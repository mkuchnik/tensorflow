# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Plumber pipeline debugging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.framework import input_pipeline_analysis_pb2
from tensorflow.python.util.tf_export import tf_export

DEBUG = True


def log_issue(*argv, **kwargs):
  if DEBUG:
    print(*argv, **kwargs)
  else:
    pass


@tf_export("data.experimental.analysis.PlumberPerformanceModel", v1=[])
class PlumberPerformanceModel(object):
  """Plumber performance model.

  Consumes a Plumber proto file.

  Link performance data to a graph structure of the pipeline. This allows
  making inferences on slow components of the pipeline.
  """

  def __init__(self, filename, use_CPU_time=False):
    self.rates_filename = filename  # Plumber stats protobuf filename
    self.plumber_data = self.get_plumber_data()  # Pipeline stats
    self.root = self.get_complete_tree(
        use_CPU_time)  # Interface for in-memory tree

  def to_graphviz(self, dot_out_filename):
    return self.root.to_graphviz(dot_out_filename)

  def mean_iter_duration(self):
    return float(self.plumber_data.iter_stats.avg_duration)

  def rate(self):
    return 1e6 / self.mean_iter_wallclock_duration()

  def var_iter_duration(self):
    return float(self.plumber_data.iter_stats.var_duration)

  def mean_iter_wallclock_duration(self):
    return float(self.plumber_data.iter_stats.avg_wallclock_duration)

  def CPU_Util(self):
    return self.cores_used() / self.cores_avail()

  def Disk_Util(self):
    return self.disk_bandwidth() / max(self.disk_bandwidth_avail(), 1e-3)

  def cores_used(self):
    return self.root.input[0].total_pipeline_time / self.root.input[0].abs_time

  def cores_avail(self):
    return self.plumber_data.machine_info.num_cores

  def mem_free(self):
    return self.plumber_data.machine_info.memory_free

  def mem_used(self):
    return self.mem_avail() - self.mem_free()

  def mem_avail(self):
    return self.plumber_data.machine_info.memory_total

  def mem_util(self):
    return self.mem_used() / self.mem_avail()

  def threads_avail(self):
    return self.plumber_data.ctx_info.shared_threadpool_size

  def udf_threads_avail(self):
    return self.plumber_data.ctx_info.udf_threadpool_size

  def bytes_read(self):
    total = 0
    for s in self.plumber_data.stats:
      if "TFRecordDataset" in s.name:
        total += s.bytes_produced
    return total

  def disk_bandwidth(self):
    return self.bytes_read() / self.total_time()

  def total_time(self):
    return self.root.input[0].abs_time / 1e9

  def disk_bandwidth_avail(self):
    return self.plumber_data.machine_info.estimated_disk_bandwidth

  def get_recommendation(self):
    return PlumberRecommendation(self)

  def to_tree(self, return_lookup=False):
    """Creates a graphdef of the dataset and returns a tree of TreeNodes."""

    graph_def = self.plumber_data.graph

    def is_important_node(node_name):
      # Remove constants nodes
      return "Dataset" in node_name or "dataset" in node_name

    def is_nested_node(node):
      # Checks for nested function in dict node.attr
      return "f" in node.attr

    # Name to node
    tree_node_lookup = dict()

    nested_nodes = dict()
    for node in graph_def.node:
      if is_important_node(node.name):
        n = TreeNode(node.name, node.op)
        n.performance_model = self
        tree_node_lookup[node.name] = n
        inputs = node.input
        for input_name in inputs:
          if is_important_node(input_name):
            input_node = tree_node_lookup[input_name]
            n.add_input(input_node)
        if is_nested_node(node):
          f_name = node.attr["f"].func.name
          nested_nodes[node.name] = f_name

    log_issue("Nested nodes", nested_nodes)

    def find_fs_of_f(f):
      """Find nested function nodes e.g., f1 calls f2."""
      fs_nodes = []
      for node in f.node_def:
        if is_nested_node(node):
          fs_nodes.append(node)
      return fs_nodes

    def find_datasets_in_f(f_name, datasets=None):
      if datasets is None:
        datasets = []
      for f in graph_def.library.function:
        if f.signature.name == f_name:
          for node in f.node_def:
            if is_important_node(node.name):
              datasets.append(node)
          child_f_nodes = find_fs_of_f(f)
          # Descend
          for child_node in child_f_nodes:
            # TFRecordDataset is node_def
            child_f_name = child_node.attr["f"].func.name
            find_datasets_in_f(child_f_name, datasets)
      return datasets

    for dataset_name, f_name in nested_nodes.items():
      datasets = find_datasets_in_f(f_name)
      dataset_names = list(map(lambda x: x.name, datasets))
      dataset_ops = list(map(lambda x: x.op, datasets))
      log_issue("Adding datasets {} to {}".format(dataset_names, dataset_name))
      new_nodes = list(
          map(lambda x: TreeNode(*x), zip(dataset_names, dataset_ops)))
      for n in new_nodes:
        n.performance_model = self
      tree_node_lookup[dataset_name].f = new_nodes
      for new_dataset_name, new_node in zip(dataset_names, new_nodes):
        tree_node_lookup[new_dataset_name] = new_node

    if return_lookup:
      return tree_node_lookup["dataset"], tree_node_lookup
    else:
      return tree_node_lookup["dataset"]

  def get_graph(self):
    """Graph def optimizations are not in sync with Python, so we read them."""
    plumber_data = self.get_plumber_data()
    graph = plumber_data.graph
    return graph

  def get_plumber_data(self):
    with open(self.rates_filename, "rb") as f:
      lines = f.read()
      plumber_data = input_pipeline_analysis_pb2.PipelineSnapshot().FromString(
          lines)
      return plumber_data

  def get_rates(self):
    rates_dict = {}
    for s in self.plumber_data.stats:
      # Note: processing_time is summed. Use count to avg.
      rates_dict[s.name] = (s.elements_produced, s.processing_time)
    return rates_dict

  def get_cpu_rates(self):
    rates_dict = {}
    for s in self.plumber_data.stats:
      # Note: processing_time is summed. Use count to avg.
      rates_dict[s.name] = (s.elements_produced, s.processing_time_clock)
    return rates_dict

  def get_abs_rates(self):
    rates_dict = {}
    for s in self.plumber_data.stats:
      # Note: wallclock_time is summed. Use count to avg.
      if s.count != 1:
        log_issue("{} has count {}".format(s.name, s.count))
      rates_dict[s.name] = (s.elements_produced, s.wallclock_time / s.count)
    return rates_dict

  def get_read_rates(self):
    rates_dict = {}
    for s in self.plumber_data.stats:
      # Note: wallclock_time is summed. Use count to avg.
      if s.count != 1:
        log_issue("{} has count {}".format(s.name, s.count))
        log_issue(s.name, "bytes", s.bytes_produced, "time",
                  s.wallclock_time / s.count)
      # TODO(mkuchnik): nested functions have multiple ephemeral iterators
      rates_dict[s.name] = (s.bytes_produced, s.wallclock_time / s.count)
    return rates_dict

  def get_ratios(self):
    rates_dict = {}
    for s in self.plumber_data.stats:
      rates_dict[s.name] = s.element_ratio
    return rates_dict

  def get_counts(self):
    rates_dict = {}
    for s in self.plumber_data.stats:
      rates_dict[s.name] = s.parallelism
    return rates_dict

  def get_size_bytes(self):
    rates_dict = {}
    for s in self.plumber_data.stats:
      rates_dict[s.name] = s.estimated_dataset_size
    return rates_dict

  def get_complete_tree(self, use_CPU_time=False):
    """Returns a tree with all members added."""

    def add_cum_ratio(node, parent_ratio, is_stale):
      """Calculates cumulative ratio and marks nodes as stale if not connected."""
      assert is_stale is not None
      if parent_ratio is None:
        node.cum_ratio = 1.0
      else:
        node.cum_ratio = parent_ratio
        node.is_stale = is_stale
      for i in node.input:
        ratio = node.ratio
        if ratio is None:
          ratio = -1.0
        child_ratio = node.cum_ratio * 1.0 / ratio
        child_is_stale = is_stale or node.is_src()
        add_cum_ratio(i, child_ratio, child_is_stale)
      if node.f:  # function members
        for i in node.f:
          ratio = node.ratio
          if ratio is None:
            ratio = -1.0
          child_ratio = node.cum_ratio * 1.0 / ratio
          child_is_stale = is_stale or node.is_src()
          add_cum_ratio(i, child_ratio, child_is_stale)

    root, lookup = self.to_tree(True)
    rates = self.get_rates()
    cpu_rates = self.get_cpu_rates()
    abs_rates = self.get_abs_rates()
    read_rates = self.get_read_rates()
    ratios = self.get_ratios()
    counts = self.get_counts()
    size_bytes = self.get_size_bytes()
    all_data = [
        rates, cpu_rates, abs_rates, read_rates, ratios, counts, size_bytes
    ]
    if any(map(lambda x: x is None, all_data)):
      return None
    for k, v in rates.items():
      if k in lookup:
        rate = -1
        if v[1]:
          rate = v[0] / v[1] * 1e9  # In nanoseconds
        lookup[k].proc_rate = rate
        lookup[k].proc_time = v[1]  # In nanoseconds
        if not use_CPU_time:
          lookup[k].rate = rate
          lookup[k].time = v[1]  # In nanoseconds
    for k, v in cpu_rates.items():
      if k in lookup:
        rate = -1
        if v[1]:
          rate = v[0] / v[1] * 1e9  # In nanoseconds
        lookup[k].cpu_rate = rate
        lookup[k].cpu_time = v[1]  # In nanoseconds
        if use_CPU_time:
          lookup[k].rate = rate
          lookup[k].time = v[1]  # In nanoseconds
    for k, v in abs_rates.items():
      if k in lookup:
        rate = -1
        if v[1]:
          rate = v[0] / v[1] * 1e9  # In nanoseconds
        lookup[k].abs_rate = rate
        lookup[k].abs_time = v[1]  # In nanoseconds
    for k, v in read_rates.items():
      if k in lookup:
        rate = -1
        if v[1]:
          rate = v[0] / v[1] * 1e9  # In nanoseconds
        lookup[k].read_rate = rate
    for k, v in ratios.items():
      if k in lookup:
        lookup[k].ratio = v
    for k, v in counts.items():
      if k in lookup:
        lookup[k].count = v
    for k, v in size_bytes.items():
      if k in lookup:
        lookup[k].size_bytes = v
    # Can scale to dataset time if ratio available
    if root.ratio is not None:
      add_cum_ratio(root, None, False)
    else:
      for i in root.input:
        add_cum_ratio(i, None, False)
    total_pipeline_time = root.total_time()  # Wait for time, is_stale to be set
    for v in lookup.values():
      v.total_pipeline_time = total_pipeline_time
      v.plumber_data = self.plumber_data
    for i in root:  # Patch functions
      if i.f:
        for j in i.f:
          # old_count = j.count
          # num = j.abs_rate * old_count
          j.count = i.count
          j.abs_time = i.abs_time
          # j.abs_rate = num / i.abs_time
          j.abs_rate = i.abs_rate
    return root


class PlumberRecommendation:

  def __init__(self, plumber_model):
    self.plumber_model = plumber_model

  def slowest_node(self):
    return self.plumber_model.root.slowest_node()

  def current_rate(self, return_debug=False):
    slow_node = self.slowest_node()
    rate = slow_node.expected_observed_rate()
    debug_str = slow_node.name
    if return_debug:
      return rate, debug_str
    else:
      return rate

  def fix_rate(self, return_debug=False, n_iterations=1):
    """Returns new rate and the new bottleneck"""
    slow_node = self.plumber_model.root.slowest_node()
    ignore_set = set([slow_node])
    second_node = self.plumber_model.root.slowest_node(ignore_set=ignore_set)

    def get_rate(slow_node, second_node, return_debug):
      predicted_rate = None
      debug_str = None
      theoretical_rate = slow_node.max_possible_rate()
      fix_rate = second_node.cum_rate()
      if theoretical_rate is not None and fix_rate is not None:
        if fix_rate < theoretical_rate:
          predicted_rate = fix_rate
          debug_str = second_node.name
        else:
          predicted_rate = theoretical_rate
          debug_str = slow_node.name
      if return_debug:
        return predicted_rate, debug_str
      else:
        return predicted_rate

    if n_iterations == 1:
      ret = get_rate(slow_node, second_node, return_debug)
      return ret
    else:
      raise NotImplementedError("fix")

  def check_max_threading(self):
    return self.plumber_model.threads_avail() == self.plumber_model.cores_avail(
    )

  def check_saturation(self):
    return self.plumber_model.CPU_Util() > 0.8 or self.plumber_model.Disk_Util(
    ) > 0.8

  def check_memory(self):
    return self.plumber_model.mem_util() > 0.8

  def upper_bounds(self):
    max_rate = 9999e10
    for n in self.plumber_model.root:
      rate = n.max_possible_rate()
      if rate is not None:
        max_rate = min(max_rate, rate)
      if n.f:
        for j in n.f:
          rate = j.max_possible_rate()
          if rate is not None:
            max_rate = min(max_rate, rate)
    return max_rate

  def actual_rate(self):
    return self.plumber_model.rate()

  def analysis_str(self):
    dbg_strs = []
    dbg_strs.append("UDF threads: {}".format(
        self.plumber_model.udf_threads_avail()))
    if not self.check_max_threading():
      dbg_strs.append("Thread and core mismatch ({}/{})".format(
          self.plumber_model.threads_avail(), self.plumber_model.cores_avail()))
    if not self.check_saturation():
      dbg_strs.append("CPU and Disk not saturated ({:.2%}/{:.2%})".format(
          self.plumber_model.CPU_Util(), self.plumber_model.Disk_Util()))
    if not self.check_memory():
      dbg_strs.append("Memory not utilized ({:.2%}) -- consider cache".format(
          self.plumber_model.mem_util()))
    rate = self.actual_rate()
    bounds = self.upper_bounds()
    efficiency = rate / bounds
    dbg_strs.append(
        "Max rate is {:.3} batches/second ({:.2}/{:.2}={:.2%} efficiency)"
        .format(bounds, rate, bounds, efficiency))
    return dbg_strs


class TreeNode:
  """Maps a running graphdef to statistics about nodes.

  Represents tensorflow graphdef nodes with their current runtime
  statistics. Names are runtime names (not type of node).
  """

  def __init__(self, name, op):
    self.name = name
    self.input = []  # Input nodes into this node
    self.rate = None  # Processing time rate
    self.abs_rate = None  # Total time rate
    self.ratio = None  # Conversion ratio between nodes (e.g., batch size)
    self.count = None  # Number of instances of this node
    self.time = None  # Time in nanoseconds give to this node
    self.abs_time = None
    self.total_pipeline_time = None  # The total time spent in pipeline
    self.cum_ratio = None  # Scaled relative to root rate
    self.f = None  # Utilized function
    self.op = op  # The graphdef node op name (e.g., type)
    self.max_rate = None  # TODO(mkuchnik): Benchmarked max rate
    self.dataset_cardinality_bytes = None  # TODO(mkuchnik): Get cache size
    self.is_stale = False  # if a src precedes the op
    self.read_rate = None
    self.plumber_data = None
    self.cpu_rate = None
    self.cpu_time = None
    self.size_bytes = None
    self.proc_rate = None
    self.proc_time = None

  def resource_type(self):
    """CPU, disk, etc."""
    if self.op == "TFRecordDataset":
      return "disk"
    else:
      return "CPU"

  def add_input(self, x):
    self.input.append(x)

  def __str__(self):
    if not self.input:
      return str(self.name)
    else:
      input_strs = []
      for i in self.input:
        input_strs.append(str(i))
      input_str = ",".join(input_strs)
      return str(self.name) + "(" + input_str + ")"

  def __repr__(self):
    """Long string."""
    content_dict = self.content_dict()
    self_repr = str(self.name) + ":" + str(content_dict)
    if not self.input:
      return self_repr
    else:
      input_strs = []
      for i in self.input:
        input_strs.append(repr(i))
      input_str = ",".join(input_strs)
      return self_repr + "(" + input_str + ")"

  def pipeline_time_percentage(self):
    if self.time and self.total_pipeline_time and not self.is_stale:
      return self.time / self.total_pipeline_time
    else:
      return None

  def is_src(self):
    """Whether this node is a source node."""
    # TODO(mkuchnik) Make this more robust
    return (self.ratio == 0.0 or self.op == "CacheDataset" or
            self.op == "CacheDatasetV2" or self.op == "TFRecordDataset")

  def is_parallelizable(self):
    """Whether this node can use more than 1 thread."""
    # TODO(mkuchnik) Make this more robust
    return (self.op == "ParallelMapDatasetV2" or
            self.op == "ParallelInterleaveDatasetV4" or
            self.op == "MapAndBatchDataset")

  def rate_slowdown(self):
    """The rate drop relative to parents.

    2 is a 2x slowdown from parent to current node.
    """
    r1 = self.actual_observed_rate()
    if self.input and r1:
      slowdowns = []
      for i in self.input:
        r2 = i.actual_observed_rate()
        if r2 is None:
          return None
        slowdown = r2 / r1  # Assume r2 >= r1
        slowdowns.append(slowdown)
      # Slowdown should be relative to slowest input
      total_slowdown = min(slowdowns)
      return total_slowdown
    else:
      return None

  def max_parallelism(self):
    """The max number of threads this node can effectively use."""
    if self.is_parallelizable():
      num_cores = self.plumber_data.machine_info.num_cores
      if self.op == "TFRecordDataset":
        disk_util_per_thread = self.Disk_util() / self.count
        max_threads = 1 / disk_util_per_thread
        return max(int(max_threads) + 1, num_cores)
      else:
        return num_cores
    else:
      return 1

  def max_possible_rate(self, discount_cores=True):
    """The maximum possible rate if maximum parallelism was used."""
    parallelism = self.max_parallelism()
    if (self.is_stale or self.rate is None or self.cum_ratio is None or
        parallelism is None):
      return None
    else:
      cores_avail = parallelism
      if discount_cores:
        cores_used = (self.total_pipeline_time - self.time) / self.abs_time
        cores_avail = max(self.count, parallelism - cores_used)
        log_issue("cores_avail for {}: {}-{}={}".format(self.name, parallelism,
                                                        cores_used,
                                                        cores_avail))
        assert cores_avail <= parallelism * self.count
      scaled_rate = self.rate * self.cum_ratio * cores_avail
      return scaled_rate

  def CPU_util(self):
    if (self.time is None or self.abs_time is None):
      return None
    return (self.time /
            self.abs_time) / self.plumber_data.machine_info.num_cores

  def Disk_util(self):
    if self.op == "TFRecordDataset":
      bw = self.performance_model.disk_bandwidth()
      bw_avail = self.performance_model.disk_bandwidth_avail()
      return bw / bw_avail
    else:
      return 0.

  def p_busy_pipeline(self):
    """How often this op is executing."""
    # Alternatively, can view scheduled slots
    # over parallelism * wallclock_time
    if self.time and self.count and self.abs_time:
      sched_time_per_op = self.time / self.count
      return sched_time_per_op / self.abs_time
    else:
      return None

  def p_busy_pipeline_single_thread(self):
    """How often this op is executing if using one core.

        This is useful to compare multithreaded ops vs. single threaded ops.
        For instance, if this is greater than 100%, then the op is definitely
        multithreaded and utilizing at least 1 core of time per time-slice.
        Note: for multithreaded ops, this can be greater than 1.
    """
    # Alternatively, can view scheduled slots
    # over parallelism * wallclock_time
    if self.time and self.abs_time:
      sched_time_per_op = self.time
      return sched_time_per_op / self.abs_time
    else:
      return None

  def content_dict(self, add_inputs=False, add_name=False):
    """Gets data out of node into dict.

        This method is useful for pretty printing.
    """
    content_dict = {
        "self_abs_rate":
            self.abs_rate,
        "self_expected_observed_rate":
            self.expected_observed_rate(),
        "self_actual_observed_rate":
            self.actual_observed_rate(),
        "self_rate":
            self.rate,
        "op":
            self.op,
        "inherited_ratio":
            self.cum_ratio,
        "scaled_rate":
            self.cum_rate(),
        "self_ratio":
            self.ratio,
        "self_op_count":
            self.count,
        "self_abs_time_s":
            self.abs_time / 1e9 if self.abs_rate else self.abs_rate,
        "self_time_ns":
            self.time,
        "self_proc_time_ns":
            self.proc_time,
        "resource_type":
            self.resource_type(),
        "pipeline_CPU_percentage":
            self.pipeline_time_percentage(),
        "p_busy":
            self.p_busy_pipeline(),
        "p_busy_single_thread":
            self.p_busy_pipeline_single_thread(),
        "is_stale":
            self.is_stale,
        "max_parallelism":
            self.max_parallelism(),
        "max_possible_rate":
            self.max_possible_rate(),
        "rate_slowdown":
            self.rate_slowdown(),
        "read_rate":
            self.read_rate,
        "self_cpu_rate":
            self.cpu_rate,
        "self_proc_rate":
            self.proc_rate,
        "self_cpu_time_ns":
            self.cpu_time,
        "self_size_bytes":
            self.size_bytes,
    }
    if self.f:
      content_dict["f"] = self.f
    if add_name:
      content_dict["name"] = self.name
    if add_inputs:
      inputs = []
      for i in self.input:
        inputs.append(i.content_dict(add_inputs, add_name))
      content_dict["inputs"] = inputs
    return content_dict

  def to_graphviz_str(self):
    """Returns a graphviz representation of the tree."""
    graphviz_str = "digraph dataset_rates {\n"
    graphviz_str += "    rankdir=LR\n"
    graphviz_str += "    size=\"8,5\"\n"
    graphviz_str += "    node [shape = circle];\n"
    shape_info_str = ""
    slowest_node = self.slowest_node()
    for i in self.__iter__():
      dst = i.name.replace("/", "")
      for j in i.input:
        src = j.name.replace("/", "")
        edge = (j.actual_observed_rate(), j.cum_rate(), j.max_possible_rate())
        if any(map(lambda x: x is None, edge)):
          graphviz_str += "    {} -> {} [ label = \"{}\" ];\n".format(
              src, dst, None)
        else:
          graphviz_str += ("    {} -> {} [ label = \""
                           "obs:{:.2e}\\n"
                           "cur_max:{:.2e}\\n"
                           "theor_max:{:.2e}\\n"
                           "\" ];\n").format(src, dst, *edge)
      if i.is_src():
        shape_info_str += "    {} [shape=Mdiamond];\n".format(dst)
      if i.is_stale:
        shape_info_str += "    {} [color=gray];\n".format(dst)
      if (i.pipeline_time_percentage() is not None and
          i.p_busy_pipeline() is not None and i.rate_slowdown() is not None):
        pipe_perc = "{:.2%}".format(i.CPU_util())
        busy_perc = "{:.2%}".format(i.p_busy_pipeline())
        slowdown = "{:.2%}".format(i.rate_slowdown())
        parallelism = i.count
        max_parallelism = i.max_parallelism()
        shape_info_str += (
            "    {dst} [label=\"{dst}(\\nCPU_util: {util},\\nbusy: "
            "{busy},\\nslowdown: {slowdown},\\nparallelism: "
            "{parallelism}/{max_parallelism})\"];\n").format(
                dst=dst,
                util=pipe_perc,
                busy=busy_perc,
                slowdown=slowdown,
                parallelism=parallelism,
                max_parallelism=max_parallelism,
            )
      else:
        shape_info_str += "    {dst} [label=\"{dst}\"];\n".format(dst=dst)
      if i.f:
        dst = i.name.replace("/", "")
        graphviz_str += "    subgraph cluster_{} {{\n".format(dst + "_func")
        graphviz_str += "        node [style=filled];\n"
        for j in i.f:
          # TODO(mkuchnik): Not recursive
          src = j.name.replace("/", "")
          edge = j.cum_rate()
          edge = (j.actual_observed_rate(), j.cum_rate(), j.max_possible_rate())
          if any(map(lambda x: x is None, edge)):
            graphviz_str += "        {} -> {} [ label = \"{}\" ];\n".format(
                src, dst, None)
          else:
            graphviz_str += ("       {} -> {} [ label = \""
                             "obs:{:.2e}\\n"
                             "cur_max:{:.2e}\\n"
                             "theor_max:{:.2e}\\n"
                             "\" ];\n").format(src, dst, *edge)
          if j.is_src():
            shape_info_str += "    {} [shape=Mdiamond];\n".format(src)
          if j.is_stale:
            shape_info_str += "    {} [color=gray];\n".format(src)
          elif j.Disk_util() > 0.75:
            shape_info_str += "    {} [color=orange];\n".format(src)
          if (j.CPU_util() is not None and j.p_busy_pipeline() is not None):
            pipe_perc = "{:.2%}".format(j.CPU_util())
            busy_perc = "{:.2%}".format(j.p_busy_pipeline())
            parallelism = j.count
            max_parallelism = j.max_parallelism()
            if j.is_src() and j.Disk_util():
              #bw = j.read_rate
              # TODO(mkuchnik): Currently, performance model more accurate
              bw = self.performance_model.disk_bandwidth()
              bw_avail = self.performance_model.disk_bandwidth_avail()
              if bw is not None:
                bw = "{:.2}/{:.2} MB/s={:.2%}".format(bw / 1e6, bw_avail / 1e6,
                                                      bw / bw_avail)
              shape_info_str += (
                  "    {dst} [label=\"{dst}(\\nCPU_util: {util},\\nDisk_BW: "
                  "{bw}\\nbusy: {busy},\\nparallelism: "
                  "{parallelism}/{max_parallelism})\"];\n").format(
                      dst=src,
                      util=pipe_perc,
                      bw=bw,
                      busy=busy_perc,
                      parallelism=parallelism,
                      max_parallelism=max_parallelism,
                  )
            else:
              shape_info_str += (
                  "    {dst} [label=\"{dst}(\\nCPU_util: {util},\\nbusy: "
                  "{busy},\\nparallelism: "
                  "{parallelism}/{max_parallelism})\"];\n").format(
                      dst=src,
                      util=pipe_perc,
                      busy=busy_perc,
                      parallelism=parallelism,
                      max_parallelism=max_parallelism,
                  )
          else:
            if j.is_src() and j.Disk_util():
              #bw = j.read_rate
              # TODO(mkuchnik): Currently, performance model more accurate
              bw = self.performance_model.disk_bandwidth()
              bw_avail = self.performance_model.disk_bandwidth_avail()
              if bw is not None:
                bw = "{:.2}/{:.2} MB/s={:.2%}".format(bw / 1e6, bw_avail / 1e6,
                                                      bw / bw_avail)
              label = "{dst}(\\Disk_BW: {bw})".format(dst=src, bw=bw)
            else:
              label = "{dst}".format(dst=src)
            shape_info_str += "    {src} [label=\"{label}\"];\n".format(
                src=src, label=label)

        graphviz_str += "        label = \"{}\";\n".format(dst + "_func")
        graphviz_str += "        color=blue\n"
        graphviz_str += "    }\n"

    graphviz_str += "\n"
    graphviz_str += "    {} [shape=Msquare];\n".format(
        self.name.replace("/", ""))
    graphviz_str += "    {} [color=red];\n".format(
        slowest_node.name.replace("/", ""))
    graphviz_str += shape_info_str
    graphviz_str += "}\n"
    return graphviz_str

  def to_graphviz(self, filename):
    """Writes graphviz to file."""
    g_str = self.to_graphviz_str()
    with open(filename, "w") as f:
      f.write(g_str)

  def full_content_dict(self):
    """Shortcut to pretty print."""
    return self.content_dict(True, True)

  def __iter__(self):
    """Iterates self and children."""
    yield self
    for i in self.input:
      for j in i.__iter__():
        yield j

    raise StopIteration

  def total_time(self, filter_f=None, return_node_times=False):
    """Returns time in this node and all children."""
    all_nodes = []
    for i in self.__iter__():
      all_nodes.append(i)
      if i.f:
        for ii in i.f:
          all_nodes.append(ii)  # TODO(mkuchnik): Not recursive
    if filter_f:
      all_nodes = filter(filter_f, all_nodes)
    node_times = list(
        map(lambda x: x.time if (x.time and not x.is_stale) else 0, all_nodes))
    total_time = np.sum(node_times)
    if return_node_times:
      return total_time, node_times
    else:
      return total_time

  def cpu_total_time(self):
    return self.total_time(lambda x: x.resource_type() == "CPU")

  def disk_total_time(self):
    return self.total_time(lambda x: x.resource_type() == "disk")

  def resource_total_time_breakdown(self, normalize=True):
    breakdown = {
        "CPU": self.cpu_total_time(),
        "disk": self.disk_total_time(),
    }
    if normalize:
      total_time = sum(filter(lambda x: x is not None, breakdown.values()))
      for k in breakdown:
        breakdown[k] = breakdown[k] / total_time
    return breakdown

  def slowest_node(self,
                   return_percent_time=False,
                   ignore_stale_nodes=True,
                   ignore_set=None):
    """Returns the slowest node.

       Set return_percent_time to return time spent.
    """
    if ignore_set is None:
      ignore_set = set()
    all_nodes = list(self.__iter__())
    if ignore_stale_nodes:
      all_nodes = list(
          filter(lambda x: not x.is_stale and x not in ignore_set, all_nodes))
    node_rates = list(map(lambda x: x.cum_rate(), all_nodes))
    min_rate = node_rates[0]
    min_rate_idx = 0
    if return_percent_time:
      total_time, node_times = self.total_time(return_node_times=True)
    for i, n in enumerate(node_rates[1:]):
      # TODO(mkuchnik): Hack to ignore ops with 0 rate
      if n == 0:
        log_issue("Ignoring node {}".format(all_nodes[i]))
        n = None
      if min_rate == 0:
        min_rate = None
      if n is not None and (min_rate is None or n < min_rate):
        min_rate = n
        min_rate_idx = i + 1
    if return_percent_time:
      return (all_nodes[min_rate_idx], node_times[min_rate_idx] / total_time)
    else:
      return all_nodes[min_rate_idx]

  def expected_observed_rate(self):
    """The rate that was observed in practice in terms of wallclock time.

        Cumulative rate only gives maximum rate if all cores were allocated
        to that node. However, there is still scheduling and queue blocking
        occurring that reduces this time by a factor of p_busy in [0,1].
        Note: Units are still normalized.
        """
    scaled_rate = self.cum_rate()
    p_busy = self.p_busy_pipeline()
    if scaled_rate is not None and p_busy is not None:
      scaled_rate_pipeline = scaled_rate * p_busy
      return scaled_rate_pipeline
    else:
      return None

  def actual_observed_rate(self):
    # For instance, parallel instances may be summed.
    if self.abs_rate is not None and self.cum_ratio is not None:
      return self.abs_rate * self.cum_ratio
    else:
      return None

  def cum_rate(self, use_cpu_time=None):
    """Return cumulative rate (scaled rate relative to root).

        cumulative_rate = processing_rate
                          * pipeline_output_ratio
                          * node_parallelism
                          * p_busy

        where:
        processing_rate - the elements/second/core
        pipeline_output_ratio - converts elements to units of minibatches
        node_parallelism - the number of parallel executions of the op
        p_busy - fraction of scheduling time dedicated to processing this op

        Returns:
        A rate in the same units per second as the output of the pipeline
    """
    if use_cpu_time is None:
      rate = self.rate
    else:
      rate = self.proc_rate if not use_cpu_time else self.cpu_rate
    if (self.is_stale or rate is None or self.cum_ratio is None or
        self.count is None):
      return None
    else:
      scaled_rate = rate * self.cum_ratio * self.count
      return scaled_rate

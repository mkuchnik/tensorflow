/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <fstream>


#include "absl/memory/memory.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/input_pipeline_analysis.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/serialization_utils.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace {

constexpr int64 kOptimizationPeriodThresholdMs = 60 * EnvTime::kSecondsToMillis;

// Default share of available RAM that can be used by model's internal buffers.
constexpr double kRamBudgetShare = 0.5;

class ModelDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ModelDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    if (ctx->HasAttr("algorithm")) {
      int64 algorithm;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("algorithm", &algorithm));
      algorithm_ = model::AutotuneAlgorithm(algorithm);
    } else {
      algorithm_ = model::AutotuneAlgorithm::HILL_CLIMB;
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cpu_budget", &cpu_budget_));
    if (ctx->HasAttr("stats_filename")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("stats_filename", &stats_filename_));
    }
    if (cpu_budget_ == 0) {
      cpu_budget_ = port::NumSchedulableCPUs();
    }
    OP_REQUIRES(ctx, cpu_budget_ > 0,
                errors::InvalidArgument("CPU budget must be positive but is ",
                                        cpu_budget_, "."));
    ram_budget_ = kRamBudgetShare * port::AvailableRam();
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    *output = new Dataset(ctx, input, algorithm_, cpu_budget_, ram_budget_,
                          stats_filename_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            model::AutotuneAlgorithm algorithm, int64 cpu_budget,
            int64 ram_budget, const string& stats_filename)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          algorithm_(algorithm),
          cpu_budget_(cpu_budget),
          ram_budget_(ram_budget),
          stats_filename_(stats_filename) {
      input_->Ref();
      if (!stats_filename_.empty()) {
        std::vector<std::pair<string, Tensor>> input_list;
        string output_node;
        Status s = AsGraphDefMinimal(ctx, input_, &input_list, &graph_def_,
                                     &output_node);
        if (s != Status::OK()) {
          VLOG(2) << "Graphdef creation failed";
        }
      }
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::Model")});
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override { return "ModelDatasetOp::Dataset"; }

    int64 Cardinality() const override { return input_->Cardinality(); }

    Status CheckExternalState() const override {
      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
        const bool force_modeling = !dataset()->stats_filename_.empty();
        model_ = std::make_shared<model::Model>(force_modeling);
      }

      ~Iterator() override {
        // Signal the optimize thread to terminate it. We will then join that
        // thread when we delete `this->optimize_thread_`.
        mutex_lock l(mu_);
        cancelled_ = true;
        cond_var_.notify_all();
      }

      Status Initialize(IteratorContext* ctx) override {
        IteratorContext::Params params(ctx);
        params.model = model_;
        return dataset()->input_->MakeIterator(
            IteratorContext(std::move(params)), this, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        IteratorContext::Params params(ctx);
        {
          mutex_lock l(mu_);
          TF_RETURN_IF_ERROR(EnsureOptimizeThreadStarted(ctx));
          params.model = model_;
          int64 now_nanos = EnvTime::NowNanos();
          RecordInput(now_nanos);
        }
        Status s = input_impl_->GetNext(IteratorContext(std::move(params)),
                                        out_tensors, end_of_sequence);
        int64 now_nanos = EnvTime::NowNanos();
        mutex_lock l(mu_);
        RecordOutput(now_nanos);
        return s;
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        return Status::OK();
      }

     private:
      Status EnsureOptimizeThreadStarted(IteratorContext* ctx)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (!model_thread_) {
          std::shared_ptr<IteratorContext> new_ctx =
              std::make_shared<IteratorContext>(*ctx);
          model_thread_ = ctx->StartThread(
              "tf_data_model", [this, new_ctx]() { ModelThread(new_ctx); });
        }
        return Status::OK();
      }

      void write_file_os(const std::string& filename,
                         const std::string& payload) {
        std::ofstream file(filename.c_str(), std::ofstream::out);
        file << payload;
      }

      void dump_stats(int64 time_nanos,
                      const std::shared_ptr<IteratorContext>& ctx) {
        GraphDef& graphdef = model_->graph_def_;
        if (!graphdef.IsInitialized()) {
          // if the model does not have a graph_def, default to ctx graph_def.
          graphdef = dataset()->graph_def_;
        }
        if (!dataset()->stats_filename_.empty() && graphdef.IsInitialized()) {
          auto stats = model_->CollectProductionStats(time_nanos);
          PipelineSnapshot snapshot = production_stats_to_proto(stats);
          *snapshot.mutable_graph() = graphdef;
          snapshot.mutable_machine_info()->set_num_cores(
              port::NumSchedulableCPUs());
          port::MemoryInfo mem_info = port::GetMemoryInfo();
          snapshot.mutable_machine_info()->set_memory_free(mem_info.free);
          snapshot.mutable_machine_info()->set_memory_total(mem_info.total);
          double avg_dur, var_dur, avg_wall_dur, disk_bw = 0;
          {
            mutex_lock l(mu_);
            avg_dur = AverageDuration();
            var_dur = VarDuration();
            avg_wall_dur = AverageWallclockDuration();
            disk_bw = disk_bandwidth_estimate_;
          }
          snapshot.mutable_machine_info()->set_estimated_disk_bandwidth(
              disk_bw);
          snapshot.mutable_ctx_info()->set_shared_threadpool_size(
              ctx->runner_threadpool_size());

          snapshot.mutable_iter_stats()->set_avg_duration(avg_dur);
          snapshot.mutable_iter_stats()->set_var_duration(var_dur);
          snapshot.mutable_iter_stats()->set_avg_wallclock_duration(
              avg_wall_dur);
          auto payload = snapshot.SerializeAsString();
          write_file_os(dataset()->stats_filename_, payload);
        } else {
          VLOG(3) << "Missing graphdef in model" << std::endl;
        }
      }

      void estimate_dist_bandwidth(int64 bytes_read, int64 elapsed_ms)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        double elapsed_s = static_cast<double>(elapsed_ms) / 1000.;
        double curr_bandwidth_estimate = bytes_read / elapsed_s;
        double avg_bw =
            (prior_disk_bandwidth_estimate_
             + disk_bandwidth_estimate_
             + curr_bandwidth_estimate) / 3.;
        const double max_update = 5e7;
        auto delta = avg_bw - disk_bandwidth_estimate_;
        if (delta > max_update) {
          disk_bandwidth_estimate_ += max_update;
        } else if (delta < -max_update) {
          disk_bandwidth_estimate_ -= max_update / 2.;
        } else {
          disk_bandwidth_estimate_ = std::max(disk_bandwidth_estimate_, avg_bw);
        }
        prior_disk_bandwidth_estimate_ = curr_bandwidth_estimate;
      }

      int64 get_total_bytes_read() {
        static monitoring::CounterCell* bytes_counter =
            metrics::GetTFDataBytesReadCounter("TFRecord");
        auto total_bytes_read = bytes_counter->value();
        return total_bytes_read;
      }

      PipelineSnapshot production_stats_to_proto(
          const absl::flat_hash_map<string, std::shared_ptr<model::Node_Stats>>&
              stats) {
        PipelineSnapshot snapshot;
        for (auto& s : stats) {
          PipelineSnapshot::OpStats* op_stats = snapshot.add_stats();
          op_stats->set_name(s.first);
          op_stats->set_elements_produced(s.second->elements_produced);
          op_stats->set_wallclock_time(s.second->wallclock_time);
          op_stats->set_processing_time(s.second->processing_time);
          op_stats->set_parallelism(s.second->parallelism);
          op_stats->set_element_ratio(s.second->ratio);
          op_stats->set_count(s.second->count);
          if (s.first == "TFRecordDataset") {
            static monitoring::CounterCell* bytes_counter =
                metrics::GetTFDataBytesReadCounter("TFRecord");
            auto total_bytes_read = bytes_counter->value();
            op_stats->set_bytes_produced(total_bytes_read);
          } else {
            op_stats->set_bytes_produced(s.second->bytes_produced);
          }
          op_stats->set_bytes_consumed(s.second->bytes_consumed);
          op_stats->set_processing_time_clock(s.second->processing_time_clock);
          op_stats->set_estimated_dataset_size(
              s.second->estimated_dataset_size);
        }
        return snapshot;
      }

      void ModelThread(const std::shared_ptr<IteratorContext>& ctx) {
        int64 last_optimization_ms = 0;
        int64 last_dump_ms = 0;
        int64 optimization_period_ms = 10;
        const int64 dump_period_ms = 1000;
        int64 current_time_ms = EnvTime::NowMicros() / EnvTime::kMillisToMicros;
        while (true) {
          {
            mutex_lock l(mu_);
            while (!cancelled_ &&
                   last_optimization_ms + optimization_period_ms >
                       current_time_ms) {
              auto wait_ms = last_optimization_ms + optimization_period_ms -
                             current_time_ms;
              VLOG(2) << "Waiting for " << wait_ms << " ms.";
              cond_var_.wait_for(l, std::chrono::milliseconds(wait_ms));
              current_time_ms = EnvTime::NowMicros() / EnvTime::kMillisToMicros;
            }
            if (cancelled_) return;
          }
          double model_input_time;
          {
            tf_shared_lock l(mu_);
            model_input_time = SelfInputTime();
          }
          model_->Optimize(dataset()->algorithm_, dataset()->cpu_budget_,
                           dataset()->ram_budget_, /*model_input_time=*/0);
          // Exponentially increase the period of running the optimization
          // until a threshold is reached.
          if (optimization_period_ms != kOptimizationPeriodThresholdMs) {
            optimization_period_ms = std::min(optimization_period_ms << 1,
                                              kOptimizationPeriodThresholdMs);
          }
          current_time_ms = EnvTime::NowMicros() / EnvTime::kMillisToMicros;
          last_optimization_ms = current_time_ms;
          model_->FlushMetrics();
          if (last_dump_ms + dump_period_ms < current_time_ms) {
            // TODO(mkuchnik): Fix coupling with other timers
            int64 time_nanos = EnvTime::NowNanos();
            int64 bytes_read = get_total_bytes_read();
            {
              mutex_lock l(mu_);
              if (last_dump_ms) {
                int64 bytes_delta = bytes_read - bytes_read_;
                estimate_dist_bandwidth(bytes_delta,
                                        time_nanos / 1000000 - last_dump_ms);
              }
              bytes_read_ = bytes_read;
            }
            dump_stats(time_nanos, ctx);
            last_dump_ms = current_time_ms;
          }
        }
      }

      void RecordInput(int64 time_nanos) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (last_output_time_ != 0) {
          DCHECK_LE(last_output_time_, time_nanos);
          const int64 duration = time_nanos - last_output_time_;
          const int64 duration_us = duration / 1000;
          double m_k1 = AverageDuration();
          input_time_ += duration;
          num_input_events_++;
          double m_k = AverageDuration();
          var_input_time_v_ += (duration_us - m_k1) * (duration_us - m_k);
        }
        if (start_time_ == 0) {
          start_time_ = time_nanos;
        }
      }

      double AverageWallclockDuration() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        auto elapsed_time = (last_output_time_ - start_time_) / 1000.;
        return static_cast<double>(elapsed_time) / num_input_events_;
      }

      double AverageDuration() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (num_input_events_ <= 0) {
          return 0.0;
        }
        auto elapsed_time = input_time_ / 1000.;
        return static_cast<double>(elapsed_time) / num_input_events_;
      }

      double VarDuration() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (num_input_events_ <= 1) {
          return 0.0;
        }
        return static_cast<double>(var_input_time_v_) / (num_input_events_ - 1);
      }

      void RecordOutput(int64 time_nanos) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        last_output_time_ = time_nanos;
      }

      double SelfInputTime() const TF_SHARED_LOCKS_REQUIRED(mu_) {
        if (num_input_events_ == 0) {
          return 0;
        }
        return static_cast<double>(input_time_) /
               static_cast<double>(num_input_events_);
      }

      mutex mu_;
      condition_variable cond_var_;
      std::shared_ptr<model::Model> model_;
      std::unique_ptr<Thread> model_thread_ TF_GUARDED_BY(mu_);
      bool cancelled_ TF_GUARDED_BY(mu_) = false;
      std::unique_ptr<IteratorBase> input_impl_;
      int64 num_input_events_ TF_GUARDED_BY(mu_) = 0;
      int64 input_time_ TF_GUARDED_BY(mu_) = 0;
      int64 last_output_time_ TF_GUARDED_BY(mu_) = 0;
      int64 var_input_time_v_ TF_GUARDED_BY(mu_) = 0;
      int64 start_time_ TF_GUARDED_BY(mu_) = 0;
      double disk_bandwidth_estimate_ TF_GUARDED_BY(mu_) = 0.0;
      double prior_disk_bandwidth_estimate_ TF_GUARDED_BY(mu_) = 0.0;
      int64 bytes_read_ TF_GUARDED_BY(mu_) = 0;
    };

    const DatasetBase* input_;
    const model::AutotuneAlgorithm algorithm_;
    const int64 cpu_budget_;
    const int64 ram_budget_;
    const string stats_filename_;
  };

  model::AutotuneAlgorithm algorithm_;
  int64 cpu_budget_;
  int64 ram_budget_;
  std::string stats_filename_;
};

REGISTER_KERNEL_BUILDER(Name("ModelDataset").Device(DEVICE_CPU),
                        ModelDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow

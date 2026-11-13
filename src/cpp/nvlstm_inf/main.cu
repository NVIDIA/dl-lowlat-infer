/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/**
 * This program executes the LSTM inference with CUDA.
 *
 * Syntax: <binary> <model> <model_dir> <data_file> <num_sec> [<output_dir>]
 *         [--num-instances <instances>] [--base_seed <seed>]
 *         [other options...]
 *
 *  <model>: one of lstm_s, lstm_m, lstm_l
 *           Use --ping-pong flag for ping pong model (ping-pong benchmark) mode
 *  <model_dir>: the directory of the model definition as created by convert_model.py
 *  <data_file>: the input data file with samples
 *  <num_sec>: the number of seconds to execute the inference
 *  <output_dir>: the directory into which inference results and time-stamp measurements are output.
 *
 * @param argc
 * @param argv
 * @return
 */

#include <algorithm>
#include <chrono>
#include <cuda/std/optional>
#include <getopt.h>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "Benchmark.cuh"
#include "ModelDef.h"
#include "cuda_helpers.cuh"
#include "ss_gemv_impl/gemv_lstm.h"
#include "ss_gemv_impl/kernel_traits.h"
#if USE_SINGLE_STEP_RESIDENT
#include "single_step/single_step_resident.cuh"
#endif
#if GDRCOPY_AVAILABLE
#include "gdrcopy_util.h"
#endif
#include "RunDefaults.h"
#if USE_FP16_REG_L_GRAPH
#include "ss_gemv_impl/fp16_reg/gemv_lstm_fp16_reg.h"
#endif

using namespace nvidia::quant::lowlat_infer;

namespace {

std::array constexpr kArgsRequired{"model", "model_dir", "data_file", "num_sec"};

std::size_t IndexRequired(std::string const &arg) {
  auto idx = std::find(kArgsRequired.begin(), kArgsRequired.end(), arg);
  if (idx != kArgsRequired.end()) {
    return idx - kArgsRequired.begin();
  } else {
    throw std::runtime_error(std::string("No such required argument: ") + arg);
  }
}

std::array constexpr kArgsOptional{"output_dir"};

std::size_t IndexOptional(std::string const &arg) {
  auto idx = std::find(kArgsOptional.begin(), kArgsOptional.end(), arg);
  if (idx != kArgsOptional.end()) {
    return kArgsRequired.size() + idx - kArgsOptional.begin();
  } else {
    throw std::runtime_error(std::string("No such optional argument: ") + arg);
  }
}

template <typename T = std::string>
std::optional<T> Option(std::uint32_t num_args, char *const argv[], int idx_offset, std::string const &arg) {
  std::optional<T> result;
  auto const idx = idx_offset + IndexOptional(arg);
  if (idx < num_args && *argv[idx] != '\0') {
    result = T{argv[idx]};
  }
  return result;
}

ModelConfig::Algo DefaultAlgo(std::string const &model_name, std::size_t batch_size) {
  return ModelConfig::Algo::SingleStepGemv;
}

std::uint32_t DefaultWarmUp(std::string const &model_name, std::size_t batch_size) {
  auto constexpr kMaxBatchSize = 1024;
  if (model_name == "lstm_s") {
    return batch_size < kMaxBatchSize ? 1 : 2;
  } else if (model_name == "lstm_m") {
    return batch_size < kMaxBatchSize ? 2 : 3;
  } else if (model_name == "lstm_l") {
    return batch_size < kMaxBatchSize ? 3 : 4;
  }

  return 1;
}

ModelConfig::Algo ParseAlgo(std::string const &algo) {
  if (algo == "SingleStep") return ModelConfig::Algo::SingleStep;
  if (algo == "SingleStepStreaming") return ModelConfig::Algo::SingleStepStreaming;
  if (algo == "SingleStepGemv") return ModelConfig::Algo::SingleStepGemv;
  if (algo == "PingPongPersistent") return ModelConfig::Algo::PingPongPersistent;
  throw std::runtime_error(algo + " is not a supported algorithm");
}

std::string PrintAlgo(ModelConfig::Algo algo) {
  if (algo == ModelConfig::Algo::SingleStep) return "SingleStep";
  if (algo == ModelConfig::Algo::SingleStepStreaming) return "SingleStepStreaming";
  if (algo == ModelConfig::Algo::SingleStepGemv) return "SingleStepGemv";
  if (algo == ModelConfig::Algo::PingPongPersistent) return "PingPongPersistent";
  throw std::runtime_error("Unknown algorithm");
}

Benchmark::SamplerType ParseSamplerType(std::string_view sType) {
  if (sType == "Random") return Benchmark::SamplerType::Random;
  StateCheck(sType == "Ring");
  return Benchmark::SamplerType::Ring;
}

std::string_view PrintSamplerType(Benchmark::SamplerType type) {
  if (type == Benchmark::SamplerType::Random) return "Random";
  StateCheck(type == Benchmark::SamplerType::Ring);
  return "Ring";
}

ssgemv::Data_kind ParsePrecision(std::string const &precision) {
  if (precision == "32") return ssgemv::DATA_TYPE_FP32;
  if (precision == "16") return ssgemv::DATA_TYPE_FP16;
  throw std::runtime_error(precision + " is not a supported precision");
}

std::chrono::nanoseconds ParseDuration(std::string const &duration) {
  std::stringstream ss{duration};
  std::int64_t ticks{-1};
  std::string unit;
  ss >> ticks >> unit;
  if (ticks < 0) throw std::runtime_error("invalid number of ticks: " + std::to_string(ticks));
  if (ticks == 0) return std::chrono::nanoseconds{0};
  auto constexpr one_ns = std::chrono::nanoseconds{1};
  if (unit == "ns") return ticks * one_ns;
  if (unit == "us") return 1'000ul * ticks * one_ns;
  if (unit == "ms") return 1'000'000ul * ticks * one_ns;
  if (unit == "s") return 1'000'000'000ul * ticks * one_ns;
  throw std::runtime_error(unit + " is not a supported unit");
}

int quiet = 0;

struct option constexpr kLongOptions[] = {
    {"algo", required_argument, nullptr, 'a'},
    {"batch-size", required_argument, nullptr, 'b'},
    {"warm-up", required_argument, nullptr, 'w'},
    {"precision", required_argument, nullptr, 'p'},
    {"grace-period", required_argument, nullptr, 'g'},
    {"ts-offset", required_argument, nullptr, 'o'},
    {"reserve", required_argument, nullptr, 'v'},
    {"cpuset", required_argument, nullptr, 'c'},
    {"clone", required_argument, nullptr, 'n'},
    {"stage", optional_argument, nullptr, 't'},
    {"reap", optional_argument, nullptr, 'r'},
    {"log-sec", optional_argument, nullptr, 'l'},
    {"use-omp", optional_argument, nullptr, 'u'},
    {"pre-compute", optional_argument, nullptr, 'k'},
    {"quiet", no_argument, &quiet, 'q'},
    {"sampler-type", required_argument, nullptr, 'z'},
    {"gdrcopy", optional_argument, nullptr, 'd'},
    {"base-seed", required_argument, nullptr, 'B'},
    {"num-instances", required_argument, nullptr, 'N'},
    {"ping-pong", no_argument, nullptr, 'P'},
    {"precompute-multi-stream", no_argument, nullptr, 'Y'},
    {"lt-algo-cache", required_argument, nullptr, 'L'},
    {"precompute-resident", optional_argument, nullptr, 'E'},
    {"precompute-step-splits", required_argument, nullptr, 'S'},
    {"lstm-l-in-graph", optional_argument, nullptr, 'G'},
    {"steps-per-graph", required_argument, nullptr, 'T'},
    {0, 0, 0, 0}  // NOLINT(modernize-use-nullptr)
};

auto constexpr kShortOptions = "a:b:w:p:g:o:v:c:n:t::r::l::u::k::q::z:B:N:PYL:E::S:G::T:d::";

void PrintSyntax(std::string const &binary_name) {
  std::cout << "Syntax: " << binary_name;
  for (auto &arg : kArgsRequired) std::cout << " <" << arg << ">";
  for (auto &arg : kArgsOptional) std::cout << " [<" << arg << ">]";
  std::cout << std::endl << "Options:" << std::endl;
  for (auto const &opt : kLongOptions) {
    if (opt.name != nullptr) {
      std::cout << "  --" << opt.name << "|-" << static_cast<char>(opt.val) << std::endl;
    }
  }
}

}  // namespace

int main(int argc, char *argv[]) {
  auto const binary_name = std::filesystem::path(argv[0]).filename().string();
  // parse options
  int opt, opt_idx;
  std::optional<std::string> opt_algo, opt_batch_size{"1"}, opt_warm_up, opt_precision{"16"}, opt_reserve,
      opt_stage{"1"}, opt_reap{"1"}, opt_log_sec{"0"}, opt_cpuset{"0, 1, 2"}, opt_grace_period{"0"}, opt_ts_offset{"0"},
      opt_use_omp{"0"}, opt_clone, opt_pre_compute{"1"}, opt_base_seed{"42"}, opt_num_instances{"1"};
  bool opt_ping_pong = false;
  bool opt_precompute_multi_stream = false;
  std::string opt_lt_algo_cache;
  // unset: on where built, applicable and working (RunDefaults.h)
  using run_defaults::Choice;
  Choice opt_gdrcopy = Choice::kAuto;
  Choice opt_precompute_resident = Choice::kAuto;
  std::optional<int> opt_precompute_step_splits;
  Choice opt_lstm_l_in_graph = Choice::kAuto;
  std::optional<int> opt_steps_per_graph;  // unset or `auto`: 8 with the in-graph kernel (RunDefaults.h)
  cuda::std::optional<std::string> opt_sampler_type;
  while ((opt = getopt_long(argc, argv, kShortOptions, kLongOptions, &opt_idx)) != -1) {
    switch (opt) {
      case 'a':
        opt_algo = optarg;
        break;
      case 'b':
        opt_batch_size = optarg;
        break;
      case 'w':
        opt_warm_up = optarg;
        break;
      case 'p':
        opt_precision = optarg;
        break;
      case 'g':
        opt_grace_period = optarg;
        break;
      case 'o':
        opt_ts_offset = optarg;
        break;
      case 'v':
        opt_reserve = optarg;
        break;
      case 'c':
        opt_cpuset = optarg;
        break;
      case 'n':
        opt_clone = optarg;
        break;
      case 'q':
        quiet = 'q';
        break;
      case 't':
        opt_stage = optarg == nullptr ? "1" : optarg;
        break;
      case 'r':
        opt_reap = optarg == nullptr ? "1" : optarg;
        break;
      case 'l':
        opt_log_sec = optarg == nullptr ? "1" : optarg;
        break;
      case 'u':
        opt_use_omp = optarg == nullptr ? "0" : optarg;
        break;
      case 'k':
        opt_pre_compute = optarg == nullptr ? "1" : optarg;
        break;
      case 'z':
        opt_sampler_type = optarg;
        break;
      case 'd':
        opt_gdrcopy = run_defaults::ParseChoice(optarg, "gdrcopy");
        break;
      case 'B':
        opt_base_seed = optarg;
        break;
      case 'N':
        opt_num_instances = optarg;
        break;
      case 'P':
        opt_ping_pong = true;
        break;
      case 'Y':
        opt_precompute_multi_stream = true;
        break;
      case 'L':
        opt_lt_algo_cache = optarg;
        break;
      case 'E':
        opt_precompute_resident = run_defaults::ParseChoice(optarg, "precompute-resident");
        break;
      case 'S':
        opt_precompute_step_splits = std::stoi(optarg);
        break;
      case 'G':
        opt_lstm_l_in_graph = run_defaults::ParseChoice(optarg, "lstm-l-in-graph");
        break;
      case 'T':
        opt_steps_per_graph = std::string_view{optarg} == "auto" ? std::nullopt : std::optional{std::stoi(optarg)};
        break;
      case 0:
        StateCheck(kLongOptions[opt_idx].flag != nullptr);
        break;
      default:
        PrintSyntax(binary_name);
        return 1;
    }
  }

  auto const idx_offset = optind;
  auto const num_args = static_cast<std::uint32_t>(argc - idx_offset);
  auto const num_required = kArgsRequired.size();
  auto const num_optional = kArgsOptional.size();

  if (num_args < num_required || num_required + num_optional < num_args) {
    PrintSyntax(binary_name);
    return 1;
  }

  std::string const model_name{argv[idx_offset + IndexRequired("model")]};
  std::string const num_instances_str = opt_num_instances ? *opt_num_instances : "1";
  auto const num_instances = static_cast<std::uint32_t>(std::stoi(num_instances_str));
  std::filesystem::path const model_dir(argv[idx_offset + IndexRequired("model_dir")]);
  std::filesystem::path const data_file(argv[idx_offset + IndexRequired("data_file")]);
  auto const num_sec = static_cast<std::uint32_t>(std::stoi(argv[idx_offset + IndexRequired("num_sec")]));

  auto const output_dir = Option<std::filesystem::path>(argc, argv, idx_offset, "output_dir");
  auto const batch_size = opt_batch_size ? static_cast<std::uint32_t>(std::stoi(*opt_batch_size)) : 1;
  auto algo = opt_algo ? ParseAlgo(*opt_algo) : DefaultAlgo(model_name, batch_size);
  auto const warm_up =
      opt_warm_up ? static_cast<std::uint32_t>(std::stoi(*opt_warm_up)) : DefaultWarmUp(model_name, batch_size);
  auto const precision = ParsePrecision(*opt_precision);
  auto const grace_period = opt_grace_period ? ParseDuration(*opt_grace_period) : std::chrono::nanoseconds{0};
  auto const stage = opt_stage && opt_stage != "0";
  auto const reap = (opt_reap && opt_reap != "0") || (!opt_reap && num_instances > 1);
  auto const log_sec = opt_log_sec ? static_cast<std::uint32_t>(std::stoi((*opt_log_sec))) : 1;
  using Clock = Benchmark::Clock;
  static_assert(std::is_same_v<std::chrono::nanoseconds::rep, long>);
  auto ts_offset = opt_ts_offset
                       ? std::optional{Clock::time_point{std::chrono::nanoseconds{std::stoul(*opt_ts_offset)}}}
                       : std::nullopt;
  static_assert(std::is_same_v<std::size_t, unsigned long>);
  std::optional<std::size_t> reserve;
  if (opt_reserve) {
    reserve = std::stoul(*opt_reserve);
  }
  auto const use_omp = opt_use_omp ? std::optional{opt_use_omp != "0"} : std::nullopt;
  auto const clone = opt_clone ? static_cast<std::uint32_t>(std::stoi(*opt_clone)) : 1u;
  auto const pre_compute = opt_pre_compute && *opt_pre_compute != "0";
  Benchmark::SamplerType sampler_type =
      opt_sampler_type.transform(ParseSamplerType).value_or(Benchmark::SamplerType::Random);
  std::uint64_t const base_seed = opt_base_seed ? std::stoull(*opt_base_seed) : 42ull;

  // Device info.
  cudaDeviceProp props;
  int dev = 0;
  CHECK_CUDA(cudaGetDeviceProperties(&props, dev));

  if (!quiet) {
    std::cout << "Running " << binary_name << std::endl
              << "  model:        " << model_name << std::endl
              << "  num_instances: " << num_instances << std::endl
              << "  base_seed:    " << base_seed << std::endl
              << "  sampler_type: " << PrintSamplerType(sampler_type) << std::endl
              << "  model_dir:    " << model_dir << std::endl
              << "  data_file:    " << data_file << std::endl
              << "  num_sec:      " << num_sec << std::endl
              << "  output_dir:   " << (output_dir ? output_dir->string() : "") << std::endl
              << "  algo:         " << PrintAlgo(algo) << std::endl
              << "  batch_size:   " << batch_size << std::endl
              << "  clone:        " << clone << std::endl
              << "  warm_up:      " << warm_up << std::endl
              << "  precision:    " << *opt_precision << std::endl
              << "  stage:        " << stage << std::endl
              << "  reap:         " << reap << std::endl
              << "  log_sec:      " << log_sec << std::endl
              << "  cpuset:       " << (opt_cpuset ? *opt_cpuset : "-") << std::endl
              << "  grace_per.:   " << (opt_grace_period ? *opt_grace_period : "") << std::endl
              << "  ts_offset:    " << (opt_ts_offset ? *opt_ts_offset : "") << std::endl
              << "  reserve:      " << (opt_reserve ? *opt_reserve : "") << std::endl
              << "  device:       " << cex::DeviceInfo() << std::endl
              << "  pre_compute:  " << (opt_pre_compute ? *opt_pre_compute : "") << std::endl
              << "  precompute_multi_stream: " << (opt_precompute_multi_stream ? "Yes" : "No") << std::endl
              << "  lt_algo_cache: " << (opt_lt_algo_cache.empty() ? "-" : opt_lt_algo_cache) << std::endl
              << "  OpenMP:       " << (*use_omp ? "Yes" : "No") << std::endl;
  }

  if (!std::filesystem::exists(model_dir) || !std::filesystem::is_directory(model_dir)) {
    throw std::runtime_error(std::string("Missing model directory: ") + model_dir.string());
  }

  auto const model_dir_name = model_dir.filename().string();
  if (model_dir_name != model_name) {
    std::cerr << "WARNING: mismatching model name " << model_name << " and model directory " << model_dir_name
              << std::endl;
  }

  if (!std::filesystem::exists(data_file) || !std::filesystem::is_regular_file(data_file)) {
    throw std::runtime_error(std::string("Missing data file: ") + data_file.string());
  }

  if (output_dir && (!std::filesystem::exists(*output_dir) || !std::filesystem::is_directory(*output_dir))) {
    throw std::runtime_error(std::string("Missing output directory: ") + output_dir->string());
  }

  auto cpus = opt_cpuset ? util::ParseList<std::uint16_t>(*opt_cpuset) : util::CpuSet{};

  // In persistent case we have main thread, a reaper thread, and one thread per instance
  // One green context per instance
  if (algo == ModelConfig::Algo::SingleStepGemv || algo == ModelConfig::Algo::PingPongPersistent) {
    if (cpus.size() < num_instances + (reap ? 2 : 1)) {
      throw std::runtime_error(
          "Insufficient number of CPU cores for the given number of instances, please check your `--cpuset` and "
          "`--num-instances`");
    }
  }

  ModelConfig model_config{};
  model_config.algorithm = algo;
  model_config.data_type = precision;
  model_config.pre_compute = pre_compute;
  model_config.max_shared_mem_per_block = props.sharedMemPerBlockOptin;
  model_config.sm_count = props.multiProcessorCount;
  model_config.sm_arch = props.major * 100 + props.minor * 10;
  // Half precision runs the precompute GEMM of layers >= 1 through cublasLt; see ModelSingleStep.
  model_config.use_cublas_lt = (precision == ssgemv::DATA_TYPE_FP16);
  // Tuned cublasLt algorithms for that GEMM (nvLstmSingleStepGemmTune); without a file the heuristic picks.
  model_config.lt_algo_cache = opt_lt_algo_cache;

  // Device info.
  printf(
      "Device...: %s (SM %d.%d), %d SMs, %zu KB max shared mem per SM\n",
      props.name,
      props.major,
      props.minor,
      props.multiProcessorCount,
      props.sharedMemPerBlockOptin / 1024);

  StateCheck(model_config.max_shared_mem_per_block / 1024 >= 99);
  bool use_precompute_green_context = false;
  bool use_green_contexts = false;
  int device_id = 0;
  if (algo == ModelConfig::Algo::SingleStepGemv) {
    // is not really helping for now
    model_config.pre_compute = false;

    {  // heuristics in case spinning kernels are used
      CudaCheck(cudaGetDevice(&device_id));
      cudaDeviceProp prop{};
      CudaCheck(cudaGetDeviceProperties(&prop, device_id));
      bool is_supported = false;
      // We may tune it for perf
      int suggested_cluster_size = 16;
      if (model_name == "lstm_s") {
        is_supported = (num_instances <= 32);
      } else if (model_name == "lstm_m") {
        if (ssgemv::is_sm_heavy_arch(model_config.sm_arch)) {
          is_supported = (num_instances <= 8);
        } else {
          is_supported = (num_instances <= 16);
        }
      } else if (model_name == "lstm_l") {
        // lstm_l occupies entire device - not supported for persistent kernels with multiple instances
        is_supported = false;
      } else {
        throw std::runtime_error("Unknown model name '" + model_name + "' for persistent kernel support check");
      }

      model_config.as_supported = (is_supported && algo == ModelConfig::Algo::SingleStepGemv);
      model_config.as_suggested_cluster_size = suggested_cluster_size;
    }

    if (model_name == "lstm_m") {
#if USE_FP16_REG_KERNELS
      // FP16 lstm_m, always spinning: the FP16 register-resident kernel, if every instance's CTAs fit at once
      if (precision == ssgemv::DATA_TYPE_FP16 && model_config.as_supported) {
        model_config.fp16_reg_m = ssgemv::verify_lstm_m_fp16_reg_occupancy_reqs(static_cast<int>(num_instances));
        if (!model_config.fp16_reg_m) {
          std::cerr << "FP16 register-resident kernel: " << num_instances
                    << " instance(s) of lstm_m do not fit this device at once; the cluster kernel runs it" << std::endl;
        }
      }
#endif
      if (!model_config.fp16_reg_m) {
        StateCheck(
            ssgemv::verify_lstm_m_occupancy_reqs(
                num_instances, model_config.sm_arch, (precision == ssgemv::DATA_TYPE_FP16) ? 2 : 4));
      }
    }

    use_green_contexts = model_config.as_supported;

  }  // if (algo == ModelConfig::Algo::SingleStepGemv)

  // Use --ping-pong flag to explicitly enable ping pong model (ping-pong benchmark) mode
  bool is_ping_pong_model = opt_ping_pong;
  // Force PingPongPersistent algo when --ping-pong is set
  if (is_ping_pong_model) {
    algo = ModelConfig::Algo::PingPongPersistent;
    model_config.algorithm = algo;  // Update model_config with the new algo
  }
  model_config.ping_pong_model_persistent = (algo == ModelConfig::Algo::PingPongPersistent);
  if (is_ping_pong_model && model_config.ping_pong_model_persistent) {
    bool is_supported = (num_instances <= 32);
    model_config.as_supported = is_supported;
    use_green_contexts = model_config.as_supported;
  }
  use_precompute_green_context = model_config.as_supported;

  // The persistent models' precompute graph runs on a single stream unless asked otherwise: the forks only add
  // dependency hops when the graph is pure precompute.
  if (algo == ModelConfig::Algo::SingleStepGemv) {
    model_config.precompute_single_stream = !opt_precompute_multi_stream;
  } else if (opt_precompute_multi_stream) {
    throw std::invalid_argument("--precompute-multi-stream requires the persistent SingleStepGemv model");
  }

  // GDRCopy (README, "Running with GDRCopy"): by default on x86 for the FP16 SingleStepGemv and ping-pong models where
  // it works here.
  auto const gdrcopy = run_defaults::ResolveGdrcopy(
      opt_gdrcopy,
      GDRCOPY_AVAILABLE != 0,
#if defined(__x86_64__)
      true,
#else
      false,
#endif
      ((algo == ModelConfig::Algo::SingleStepGemv || model_config.ping_pong_model_persistent) &&
       precision == ssgemv::DATA_TYPE_FP16)
          ? ""
          : "on by default for the FP16 SingleStepGemv and ping-pong models",
      [] {
#if GDRCOPY_AVAILABLE
        auto const probe = gdrcopy_utils::probe_gdrcopy();
        return run_defaults::GdrcopyCheck{probe.ok, probe.description};
#else
        return run_defaults::GdrcopyCheck{};
#endif
      });
  model_config.use_gdrcopy = gdrcopy.on;

  // The resident SingleStep precompute (single_step/README.md): for the FP16 register kernels of the persistent lstm_s
  // and lstm_m models, one model per instance (the other models of an instance get each input through StoreInput(),
  // which it lacks).
  std::string resident_not_applicable;
  if (algo != ModelConfig::Algo::SingleStepGemv || !model_config.as_supported || precision != ssgemv::DATA_TYPE_FP16 ||
      !(model_name == "lstm_s" || (model_name == "lstm_m" && model_config.fp16_reg_m))) {
    resident_not_applicable =
        "requires the persistent FP16 SingleStepGemv model with the FP16 register kernel (lstm_s, or lstm_m where its "
        "instances fit)";
  } else if (clone > 1) {
    resident_not_applicable = "supports one model per instance (--clone=1)";
  }
  auto resident = run_defaults::Resolve(
      opt_precompute_resident,
      "precompute-resident",
      USE_SINGLE_STEP_RESIDENT != 0,
      "-DUSE_SINGLE_STEP_RESIDENT=ON",
      resident_not_applicable);
  if (opt_precompute_step_splits) {
    if (!resident.on) throw std::invalid_argument("--precompute-step-splits requires --precompute-resident");
    if (*opt_precompute_step_splits < 1) throw std::invalid_argument("--precompute-step-splits must be at least 1");
  }
  model_config.precompute_resident = resident.on;
  model_config.precompute_step_splits = opt_precompute_step_splits.value_or(1);

  // The FP16 register lstm_l kernel in its step's precompute graph (USE_FP16_REG_L_GRAPH, fp16_reg/README.md).
  std::string in_graph_not_applicable;
  if (algo != ModelConfig::Algo::SingleStepGemv || precision != ssgemv::DATA_TYPE_FP16 || model_name != "lstm_l") {
    in_graph_not_applicable = "requires the FP16 SingleStepGemv lstm_l model";
  } else {
#if USE_FP16_REG_L_GRAPH
    // the model runs the FP16 register kernel only at its sizes and if its CTAs fit the device at once
    ModelDef const model_def{model_dir.c_str()};
    ssgemv::Fp16_reg_model_dims const dims{
        int(model_def.NumFeatures()), int(model_def.NumUnits()), int(model_def.NumLayers())};
    if (!(dims == ssgemv::FP16_REG_LSTM_L_DIMS) || !ssgemv::verify_lstm_l_fp16_reg_occupancy_reqs()) {
      in_graph_not_applicable = "requires the FP16 register lstm_l kernel (its sizes, its CTAs fitting this device)";
    }
#endif
  }
  auto const in_graph = run_defaults::Resolve(
      opt_lstm_l_in_graph,
      "lstm-l-in-graph",
      USE_FP16_REG_L_GRAPH != 0,
      "-DUSE_FP16_REG_L_GRAPH=ON",
      in_graph_not_applicable);
  model_config.lstm_l_in_graph = in_graph.on;
  model_config.steps_per_graph = run_defaults::ResolveStepsPerGraph(opt_steps_per_graph, in_graph.on);

  if (!quiet) {
    std::cout << "  Green Contexts: " << (use_green_contexts ? "Yes" : "No") << std::endl;
    if (use_green_contexts)
      std::cout << "  Use a separate green context for precompute: " << (use_precompute_green_context ? "Yes" : "No")
                << std::endl;
  }

  GreenContextsVector green_contexts;

  if (use_green_contexts)
    green_contexts = CreateGreenContexts(
        model_dir,
        num_instances,
        model_config.sm_arch,
        device_id,
        use_precompute_green_context,
        is_ping_pong_model,
        model_config.fp16_reg_m && !is_ping_pong_model);

#if USE_SINGLE_STEP_RESIDENT
  if (model_config.precompute_resident) {
    // Every instance's resident precompute kernel stays resident with one CTA per SM, all of them in the precompute's
    // partition: more CTAs than its SMs could never all run, and the kernels would wait on each other for ever. One SM
    // stays free for the work an Init on running kernels issues there (clearing the buffers).
    ModelDef const model_def{model_dir.c_str()};  // its sizes come from the weights' shapes
    auto const ctas = single_step::SingleStepResident::CtasFor(
                          static_cast<int>(model_def.NumUnits()),
                          static_cast<int>(model_def.NumSteps()),
                          model_config.precompute_step_splits) *
                      static_cast<int>(num_instances);
    auto const sms = static_cast<int>(g_gctx_remaining_partition.sm.smCount);
    if (ctas >= sms) {
      resident = run_defaults::Refused(
          opt_precompute_resident,
          "precompute-resident",
          std::to_string(num_instances) + " instance(s) need " + std::to_string(ctas) +
              " SMs for their precompute and one more, the partition has " + std::to_string(sms) +
              " (fewer instances or step splits)",
          "the cuBLAS precompute runs");
      model_config.precompute_resident = false;
    }
  }
#endif

  if (!quiet) {
    std::cout << "  GDRCopy:      " << (gdrcopy.on ? "Yes" : "No") << " (" << gdrcopy.note << ")" << std::endl
              << "  precompute_resident: " << (resident.on ? "Yes" : "No") << " (" << resident.note << ")" << std::endl
              << "  precompute_step_splits: " << model_config.precompute_step_splits
              << (model_config.precompute_resident ? "" : " (unused)") << std::endl
              << "  lstm_l_in_graph: " << (in_graph.on ? "Yes" : "No") << " (" << in_graph.note << ")" << std::endl
              << "  steps_per_graph: " << model_config.steps_per_graph
              << (!opt_steps_per_graph && in_graph.on ? " (default)" : "") << std::endl;
  }

  StateCheck(model_config.as_supported && !*use_omp || !model_config.as_supported);

  Benchmark::ResultVector results{};
  {  // scope for GPU resource allocations
    Benchmark benchmark{
        data_file,
        model_dir,
        {base_seed, sampler_type},
        num_instances,
        batch_size,
        model_config,
        stage,
        reap,
        cpus,
        grace_period,
        ts_offset,
        use_omp,
        clone,
        green_contexts,
        use_precompute_green_context};
    if (stage && !quiet) {
      std::cout << "  use_omp:    " << benchmark.UseOmp() << std::endl;
    }

    if (warm_up) {
      benchmark.Run(warm_up, 0);
      auto const throughput_per_sec = benchmark.Iterations() / (warm_up);
      if (!quiet) {
        std::cout << " ~throughput: " << throughput_per_sec << " iterations per second" << std::endl;
      }
      benchmark.Reset();
      benchmark.Reserve(
          static_cast<std::size_t>(static_cast<double>(throughput_per_sec) * num_sec * 1.2 / num_instances));
    }

    if (reserve) {
      benchmark.Reserve(*reserve);
    }
    // Actual measurements
    benchmark.Run(num_sec, log_sec);
    results = benchmark.Results();
    ts_offset = benchmark.TsBase();
    benchmark.Finalize();
  }
  green_contexts.resize(0);

  StateCheck(results.size() == num_instances);

  // release GPU resources
  CudaCheck(cudaDeviceReset());

  auto const iterations =
      std::accumulate(results.begin(), results.end(), std::size_t{}, [](auto const &acc, auto const &res) {
        return acc + res->Size();
      });

  if (!quiet) {
    auto const throughput = iterations / num_sec;
    ModelDef model_def{model_dir.c_str(), model_config.ping_pong_model_persistent};
    auto const flops = static_cast<double>(model_def.Flops() * iterations) / num_sec * 1e-12;
    std::cout << "Finished" << std::endl
              << "  Throughput: " << throughput << " iterations per second" << std::endl
              << "  Compute:    " << flops << " TFLOPS per second" << std::endl
              << "  Iterations: " << iterations / num_instances << " per model" << std::endl;
  }

  if (output_dir) {
    auto const file_base = output_dir.value() / (model_name + "." + num_instances_str + ".");
    auto const suffix = num_instances > 1 ? "x.npy" : "1.npy";
    std::cout << "Saving results to " << file_base.string() << suffix << std::endl;
    Benchmark::Save(file_base, results, *ts_offset);
  }

  if (!quiet) {
    auto const duration =
        std::accumulate(results.begin(), results.end(), Clock::duration{}, [](auto const &acc, auto const &res) {
          return acc + res->Duration();
        });

    auto const latency = std::chrono::duration_cast<std::chrono::nanoseconds>(duration / iterations).count();

    auto const latency_p50 = std::chrono::duration_cast<std::chrono::nanoseconds>(results[0]->P50()).count();
    auto const latency_p99 = std::chrono::duration_cast<std::chrono::nanoseconds>(results[0]->P99()).count();

    std::cout << "  Latency:     " << latency / 1000.f << " micro-seconds" << std::endl
              << "  P50 Latency: " << latency_p50 / 1000.f << " micro-seconds" << std::endl
              << "  P99 Latency: " << latency_p99 / 1000.f << " micro-seconds" << std::endl
              << "Terminating " << binary_name << std::endl;
  }
}

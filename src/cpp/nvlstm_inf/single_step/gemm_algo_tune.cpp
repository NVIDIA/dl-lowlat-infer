/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// nvLstmSingleStepGemmTune: picks the fastest cublasLt algorithm for the input GEMM of layers >= 1 in the SingleStep
// precompute, C[4U x steps] = W[4U x U] * X[U x steps] in half precision (U = units).
//
// In half precision ModelSingleStep runs this GEMM with cublasLt, the heuristic's first pick without workspace.  This
// tool times a large candidate set, reports the winner next to that pick and cublasGemmEx and, with --cache=FILE,
// writes it to a cache file (format in lt_algo_cache.h) when it beats the model's pick.  nvLstmInf reads the file with
// --lt-algo-cache=FILE.
//
// Candidates: the cublasLt heuristic list (--candidates N) and, with --exhaustive, every (algo id, tile, stages,
// split-K, reduction scheme, swizzle) combination the library reports as valid for the shape.  All candidates get a
// quick timing, and the best few are re-timed inside a CUDA graph.
//
// --sms=N times everything on a green context of the SMs left after splitting off the device's other SMs, N when
// those are a multiple of the SM granularity (the precompute's share of the device when it runs next to persistent
// kernels; nvLstmInf prints that share with --lt-algo-cache), and keys the cache entry on that context's SM count;
// --batch=B times B GEMMs of the shape as one strided-batched call (one per layer, as the recurrent GEMM of the
// precompute), against cublasGemmStridedBatchedEx (not cached: the model does not run that call through cublasLt).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda/cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "nvlstm_inf/LstmConstants.hpp"
#include "nvlstm_inf/cuda_error_util.h"
#include "nvlstm_inf/single_step/lt_algo_cache.h"
#include "nvlstm_inf/ss_gemv_impl/kernel_params.h"

namespace {

namespace lc = nvidia::quant::lowlat_infer::lt_cache;

constexpr int GATES = static_cast<int>(nvidia::quant::lowlat_infer::kNumGates);
constexpr int GRAPH_INNER = 16;
// How many algorithm ids / structured capability entries to query.
constexpr int MAX_ALGO_IDS = 64;
constexpr int SPLITK_CHOICES[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};
// ModelSingleStep<half>'s data and compute types for this GEMM.
constexpr cudaDataType DTYPE = CUDA_R_16F;
constexpr cublasComputeType_t COMPUTE = CUBLAS_COMPUTE_16F;

struct Options {
  int units = 0;
  int steps = 0;
  int candidates = 64;
  bool exhaustive = false;
  int reps = 200;
  int refine = 8;
  int max_candidates = 20000;
  std::size_t workspace_mb = 256;
  std::string cache_path;
  int device = 0;
  int sms = 0;            // > 0: on a green context of the SMs left after splitting off all but this many
  int batch = 1;          // > 1: strided-batched, this many GEMMs per call
  int k = 0;              // > 0: K other than units (e.g. input and recurrent weights side by side, K = 2 * units)
  double min_gain = 2.0;  // %: a cache entry only for a tuned algorithm at least this much faster than the model's pick
};

struct Preset {
  char const* name;
  int steps;
  int units;
};

// The build's model sizes (ss_gemv_impl/kernel_params.h: the built-in ones, or MODEL_PARAMS_HEADER's), as nvLstmInf
// runs them.  Anything else is given with --units/--steps.
constexpr Preset PRESETS[] = {
    {"lstm_s", LSTM_S_NUM_STEPS, LSTM_S_NUM_UNITS},
    {"lstm_m", LSTM_M_NUM_STEPS, LSTM_M_NUM_UNITS},
    {"lstm_l", LSTM_L_NUM_STEPS, LSTM_L_NUM_UNITS}};

struct Candidate {
  cublasLtMatmulAlgo_t algo{};
  bool heuristic = false;  // from the heuristic list (always re-timed precisely)
  std::size_t workspace = 0;
  std::string label;
  float quick_us = 0.0f;
  float precise_us = 0.0f;
};

struct Shape {
  int m = 0;  // 4 * units
  int n = 0;  // steps
  int k = 0;  // units
};

// Everything needed to run the GEMM repeatedly.
class Problem {
 public:
  Problem(Shape const& shape, Options const& opt, cublasLtHandle_t lt, cudaStream_t stream)
      : shape_(shape), opt_(opt), lt_(lt), stream_(stream) {
    // Same padding as cex::Malloc2D: pitches are multiples of 128 bytes.
    int const pad = 128 / static_cast<int>(sizeof(__half));
    lda_ = cuda::round_up(shape.k, pad);
    ldb_ = cuda::round_up(shape.k, pad);
    ldc_ = cuda::round_up(shape.m, pad);

    CublasCheck(cublasLtMatmulDescCreate(&op_, COMPUTE, DTYPE));
    cublasOperation_t const ta = CUBLAS_OP_T;
    cublasOperation_t const tb = CUBLAS_OP_N;
    CublasCheck(cublasLtMatmulDescSetAttribute(op_, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CublasCheck(cublasLtMatmulDescSetAttribute(op_, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    CublasCheck(cublasLtMatrixLayoutCreate(&a_, DTYPE, shape.k, shape.m, lda_));
    CublasCheck(cublasLtMatrixLayoutCreate(&b_, DTYPE, shape.k, shape.n, ldb_));
    CublasCheck(cublasLtMatrixLayoutCreate(&c_, DTYPE, shape.m, shape.n, ldc_));
    stride_a_ = static_cast<long long>(lda_) * shape.m;
    stride_b_ = static_cast<long long>(ldb_) * shape.n;
    stride_c_ = static_cast<long long>(ldc_) * shape.n;
    if (opt.batch > 1) {
      for (auto [layout, stride] : {std::pair{a_, stride_a_}, std::pair{b_, stride_b_}, std::pair{c_, stride_c_}}) {
        int const count = opt.batch;
        CublasCheck(
            cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)));
        CublasCheck(cublasLtMatrixLayoutSetAttribute(
            layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
      }
    }

    std::size_t const a_bytes = static_cast<std::size_t>(stride_a_) * opt.batch * sizeof(__half);
    std::size_t const b_bytes = static_cast<std::size_t>(stride_b_) * opt.batch * sizeof(__half);
    std::size_t const c_bytes = static_cast<std::size_t>(stride_c_) * opt.batch * sizeof(__half);
    CudaCheck(cudaMalloc(&da_, a_bytes));
    CudaCheck(cudaMalloc(&db_, b_bytes));
    CudaCheck(cudaMalloc(&dc_, c_bytes));
    CudaCheck(cudaMemset(dc_, 0, c_bytes));
    Fill(da_, a_bytes);
    Fill(db_, b_bytes);
    workspace_bytes_ = opt.workspace_mb << 20;
    CudaCheck(cudaMalloc(&workspace_, workspace_bytes_));

    alpha_ = __float2half(1.0f);
    beta_ = __float2half(0.0f);
  }

  ~Problem() {
    cudaFree(da_);
    cudaFree(db_);
    cudaFree(dc_);
    cudaFree(workspace_);
    cublasLtMatrixLayoutDestroy(a_);
    cublasLtMatrixLayoutDestroy(b_);
    cublasLtMatrixLayoutDestroy(c_);
    cublasLtMatmulDescDestroy(op_);
  }

  Problem(Problem const&) = delete;
  Problem& operator=(Problem const&) = delete;

  Shape const& shape() const { return shape_; }
  cublasLtMatmulDesc_t op() const { return op_; }
  cublasLtMatrixLayout_t a_layout() const { return a_; }
  cublasLtMatrixLayout_t b_layout() const { return b_; }
  cublasLtMatrixLayout_t c_layout() const { return c_; }

  // cublasLtMatmulAlgoCheck; on success fills the workspace requirement.
  bool Check(cublasLtMatmulAlgo_t const& algo, std::size_t& workspace) const {
    cublasLtMatmulHeuristicResult_t res{};
    if (cublasLtMatmulAlgoCheck(lt_, op_, a_, b_, c_, c_, &algo, &res) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    workspace = res.workspaceSize;
    return res.workspaceSize <= workspace_bytes_;
  }

  bool Launch(cublasLtMatmulAlgo_t const& algo) const {
    return cublasLtMatmul(
               lt_,
               op_,
               &alpha_,
               da_,
               a_,
               db_,
               b_,
               &beta_,
               dc_,
               c_,
               dc_,
               c_,
               &algo,
               workspace_,
               workspace_bytes_,
               stream_) == CUBLAS_STATUS_SUCCESS;
  }

  // The cuBLAS call the model falls back to without a cache entry (strided-batched with --batch).
  bool LaunchGemmEx(cublasHandle_t cublas) const {
    if (opt_.batch > 1) {
      return cublasGemmStridedBatchedEx(
                 cublas,
                 CUBLAS_OP_T,
                 CUBLAS_OP_N,
                 shape_.m,
                 shape_.n,
                 shape_.k,
                 &alpha_,
                 da_,
                 DTYPE,
                 lda_,
                 stride_a_,
                 db_,
                 DTYPE,
                 ldb_,
                 stride_b_,
                 &beta_,
                 dc_,
                 DTYPE,
                 ldc_,
                 stride_c_,
                 opt_.batch,
                 COMPUTE,
                 CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
    }
    return cublasGemmEx(
               cublas,
               CUBLAS_OP_T,
               CUBLAS_OP_N,
               shape_.m,
               shape_.n,
               shape_.k,
               &alpha_,
               da_,
               DTYPE,
               lda_,
               db_,
               DTYPE,
               ldb_,
               &beta_,
               dc_,
               DTYPE,
               ldc_,
               COMPUTE,
               CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
  }

  // Average microseconds per call of `fn`, a handful of calls timed back to back (hot L2).  Returns <0 on failure.
  template <typename Fn>
  float QuickTime(Fn&& fn) const {
    for (int i = 0; i < 2; ++i) {
      if (!fn()) {
        return -1.0f;
      }
    }
    if (cudaStreamSynchronize(stream_) != cudaSuccess || cudaGetLastError() != cudaSuccess) {
      return -1.0f;
    }
    cudaEvent_t t0, t1;
    CudaCheck(cudaEventCreate(&t0));
    CudaCheck(cudaEventCreate(&t1));
    constexpr int N = 10;
    CudaCheck(cudaEventRecord(t0, stream_));
    for (int i = 0; i < N; ++i) {
      fn();
    }
    CudaCheck(cudaEventRecord(t1, stream_));
    CudaCheck(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CudaCheck(cudaEventElapsedTime(&ms, t0, t1));
    CudaCheck(cudaEventDestroy(t0));
    CudaCheck(cudaEventDestroy(t1));
    return ms * 1000.0f / N;
  }

  // Median over several batches, each replaying a CUDA graph of GRAPH_INNER calls (launch overhead amortised the way
  // the model's captured graph amortises it).
  template <typename Fn>
  float PreciseTime(Fn&& fn) const {
    if (QuickTime(fn) < 0.0f) {
      return -1.0f;
    }
    constexpr int BATCHES = 5;
    int const per_batch = std::max(1, opt_.reps / BATCHES);
    std::vector<float> samples;
    cudaEvent_t t0, t1;
    CudaCheck(cudaEventCreate(&t0));
    CudaCheck(cudaEventCreate(&t1));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CudaCheck(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < GRAPH_INNER; ++i) {
      fn();
    }
    CudaCheck(cudaStreamEndCapture(stream_, &graph));
    CudaCheck(cudaGraphInstantiate(&exec, graph, 0));
    for (int i = 0; i < 3; ++i) {
      CudaCheck(cudaGraphLaunch(exec, stream_));
    }
    CudaCheck(cudaStreamSynchronize(stream_));
    int const launches = std::max(1, per_batch / GRAPH_INNER);
    for (int b = 0; b < BATCHES; ++b) {
      CudaCheck(cudaEventRecord(t0, stream_));
      for (int i = 0; i < launches; ++i) {
        CudaCheck(cudaGraphLaunch(exec, stream_));
      }
      CudaCheck(cudaEventRecord(t1, stream_));
      CudaCheck(cudaEventSynchronize(t1));
      float ms = 0.0f;
      CudaCheck(cudaEventElapsedTime(&ms, t0, t1));
      samples.push_back(ms * 1000.0f / static_cast<float>(launches * GRAPH_INNER));
    }
    CudaCheck(cudaGraphExecDestroy(exec));
    CudaCheck(cudaGraphDestroy(graph));
    CudaCheck(cudaEventDestroy(t0));
    CudaCheck(cudaEventDestroy(t1));
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
  }

 private:
  void Fill(void* dev, std::size_t bytes) const {
    // Non-trivial data: all-zero operands lower tensor-core power and can flatter the timing.
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::vector<__half> host(bytes / sizeof(__half));
    for (auto& v : host) {
      v = __float2half(dist(rng));
    }
    CudaCheck(cudaMemcpy(dev, host.data(), bytes, cudaMemcpyHostToDevice));
  }

  Shape shape_;
  Options opt_;
  cublasLtHandle_t lt_;
  cudaStream_t stream_;
  int lda_, ldb_, ldc_;
  long long stride_a_ = 0, stride_b_ = 0, stride_c_ = 0;  // elements between the GEMMs of a batch
  cublasLtMatmulDesc_t op_{};
  cublasLtMatrixLayout_t a_{}, b_{}, c_{};
  void *da_ = nullptr, *db_ = nullptr, *dc_ = nullptr, *workspace_ = nullptr;
  std::size_t workspace_bytes_ = 0;
  __half alpha_, beta_;
};

std::string Describe(cublasLtMatmulAlgo_t const& algo) {
  auto get = [&](cublasLtMatmulAlgoConfigAttributes_t attr) {
    std::uint32_t v = 0;
    std::size_t written = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &v, sizeof(v), &written);
    return v;
  };
  char buf[160];
  std::snprintf(
      buf,
      sizeof(buf),
      "id=%u tile=%u stages=%u splitk=%u red=%u swz=%u opt=%u",
      get(CUBLASLT_ALGO_CONFIG_ID),
      get(CUBLASLT_ALGO_CONFIG_TILE_ID),
      get(CUBLASLT_ALGO_CONFIG_STAGES_ID),
      get(CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
      get(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME),
      get(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),
      get(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION));
  return buf;
}

template <typename T>
std::vector<T> CapArray(cublasLtMatmulAlgo_t const& algo, cublasLtMatmulAlgoCapAttributes_t attr) {
  std::size_t bytes = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(&algo, attr, nullptr, 0, &bytes) != CUBLAS_STATUS_SUCCESS || bytes == 0) {
    return {};
  }
  std::vector<T> out(bytes / sizeof(T));
  if (cublasLtMatmulAlgoCapGetAttribute(&algo, attr, out.data(), bytes, &bytes) != CUBLAS_STATUS_SUCCESS) {
    return {};
  }
  return out;
}

template <typename T>
T CapScalar(cublasLtMatmulAlgo_t const& algo, cublasLtMatmulAlgoCapAttributes_t attr) {
  T v{};
  std::size_t written = 0;
  cublasLtMatmulAlgoCapGetAttribute(&algo, attr, &v, sizeof(v), &written);
  return v;
}

template <typename T>
void SetConfig(cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr, T v) {
  cublasLtMatmulAlgoConfigSetAttribute(&algo, attr, &v, sizeof(v));
}

std::vector<Candidate> HeuristicCandidates(Problem const& p, cublasLtHandle_t lt, int count, std::size_t ws_bytes) {
  cublasLtMatmulPreference_t pref;
  CublasCheck(cublasLtMatmulPreferenceCreate(&pref));
  CublasCheck(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
  std::vector<cublasLtMatmulHeuristicResult_t> res(count);
  int n = 0;
  CublasCheck(cublasLtMatmulAlgoGetHeuristic(
      lt, p.op(), p.a_layout(), p.b_layout(), p.c_layout(), p.c_layout(), pref, count, res.data(), &n));
  cublasLtMatmulPreferenceDestroy(pref);
  std::vector<Candidate> out;
  for (int i = 0; i < n; ++i) {
    Candidate c;
    c.algo = res[i].algo;
    c.workspace = res[i].workspaceSize;
    c.label = "heur[" + std::to_string(i) + "] " + Describe(c.algo);
    c.heuristic = true;
    out.push_back(c);
  }
  return out;
}

std::vector<Candidate> ExhaustiveCandidates(Problem const& p, cublasLtHandle_t lt, int max_candidates) {
  std::vector<Candidate> out;
  int ids[MAX_ALGO_IDS];
  int num_ids = 0;
  if (cublasLtMatmulAlgoGetIds(lt, COMPUTE, DTYPE, DTYPE, DTYPE, DTYPE, DTYPE, MAX_ALGO_IDS, ids, &num_ids) !=
      CUBLAS_STATUS_SUCCESS) {
    return out;
  }
  for (int i = 0; i < num_ids; ++i) {
    cublasLtMatmulAlgo_t base{};
    if (cublasLtMatmulAlgoInit(lt, COMPUTE, DTYPE, DTYPE, DTYPE, DTYPE, DTYPE, ids[i], &base) !=
        CUBLAS_STATUS_SUCCESS) {
      continue;
    }
    auto tiles = CapArray<std::uint32_t>(base, CUBLASLT_ALGO_CAP_TILE_IDS);
    auto stages = CapArray<std::uint32_t>(base, CUBLASLT_ALGO_CAP_STAGES_IDS);
    if (stages.empty()) {
      stages.push_back(0);
    }
    bool const splitk_ok = CapScalar<std::int32_t>(base, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT) != 0;
    auto const red_mask = CapScalar<std::uint32_t>(base, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK);
    bool const swizzle_ok = CapScalar<std::int32_t>(base, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT) != 0;

    for (auto tile : tiles) {
      for (auto stage : stages) {
        for (int split : SPLITK_CHOICES) {
          if (split > 1 && !splitk_ok) {
            continue;
          }
          // A split-K slice below 32 columns of K is never useful.
          if (split > 1 && p.shape().k / split < 32) {
            continue;
          }
          std::vector<std::uint32_t> schemes;
          if (split == 1) {
            schemes.push_back(CUBLASLT_REDUCTION_SCHEME_NONE);
          } else {
            for (std::uint32_t s :
                 {CUBLASLT_REDUCTION_SCHEME_INPLACE,
                  CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE,
                  CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE}) {
              if (red_mask & s) {
                schemes.push_back(s);
              }
            }
          }
          for (auto scheme : schemes) {
            for (std::uint32_t swz = 0; swz <= (swizzle_ok ? 1u : 0u); ++swz) {
              Candidate c;
              c.algo = base;
              SetConfig<std::uint32_t>(c.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, tile);
              SetConfig<std::uint32_t>(c.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, stage);
              SetConfig<std::uint32_t>(c.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, static_cast<std::uint32_t>(split));
              SetConfig<std::uint32_t>(c.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, scheme);
              SetConfig<std::uint32_t>(c.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, swz);
              if (!p.Check(c.algo, c.workspace)) {
                continue;
              }
              c.label = "enum " + Describe(c.algo);
              out.push_back(c);
              if (static_cast<int>(out.size()) >= max_candidates) {
                return out;
              }
            }
          }
        }
      }
    }
  }
  return out;
}

double Tflops(Shape const& s, float us) { return us > 0.0f ? 2.0 * s.m * s.n * s.k / (us * 1e-6) / 1e12 : 0.0; }

struct TuneResult {
  Candidate best;
  float gemm_ex_us = 0.0f;
  float model_default_us = 0.0f;
  bool model_default_wins = false;
};

TuneResult Tune(Problem const& p, Options const& opt, cublasLtHandle_t lt, cublasHandle_t cublas) {
  auto const& s = p.shape();
  std::printf("\nM=%d N=%d K=%d (%.1f MFLOP)\n", s.m, s.n, s.k, 2.0 * s.m * s.n * s.k / 1e6);

  std::size_t const ws_bytes = opt.workspace_mb << 20;
  auto cands = HeuristicCandidates(p, lt, opt.candidates, ws_bytes);
  std::printf("heuristic candidates: %zu\n", cands.size());

  // ModelSingleStep's pick without a cache entry: the heuristic's first result without workspace.  Kept as a
  // candidate, and the bar a cache entry has to beat.
  Candidate model_default;
  {
    auto first = HeuristicCandidates(p, lt, 1, 0);
    if (first.empty()) {
      throw std::runtime_error("no cublasLt heuristic result for the shape");
    }
    model_default = first.front();
    model_default.label = "model default " + model_default.label;
  }

  if (opt.exhaustive) {
    auto more = ExhaustiveCandidates(p, lt, opt.max_candidates);
    std::printf("enumerated candidates: %zu\n", more.size());
    cands.insert(cands.end(), more.begin(), more.end());
  }

  for (auto& c : cands) {
    c.quick_us = p.QuickTime([&] { return p.Launch(c.algo); });
  }
  cands.erase(
      std::remove_if(cands.begin(), cands.end(), [](Candidate const& c) { return c.quick_us <= 0.0f; }), cands.end());
  std::sort(cands.begin(), cands.end(), [](Candidate const& a, Candidate const& b) { return a.quick_us < b.quick_us; });
  std::printf("runnable candidates: %zu\n", cands.size());
  if (cands.empty()) {
    throw std::runtime_error("no runnable cublasLt algorithm");
  }

  // The best --refine of the quick pass, and every heuristic candidate: one quick timing can rank a good algorithm
  // low (e.g. while the clocks ramp up), and the heuristic's list is short.
  std::vector<Candidate> refined;
  for (std::size_t i = 0; i < cands.size(); ++i) {
    if (i < static_cast<std::size_t>(opt.refine) || cands[i].heuristic) refined.push_back(cands[i]);
  }
  cands.swap(refined);
  for (auto& c : cands) {
    c.precise_us = p.PreciseTime([&] { return p.Launch(c.algo); });
  }
  model_default.precise_us = p.PreciseTime([&] { return p.Launch(model_default.algo); });
  // A failed precise timing (< 0) would otherwise sort first and be taken as the winner.
  cands.erase(
      std::remove_if(cands.begin(), cands.end(), [](Candidate const& c) { return c.precise_us <= 0.0f; }), cands.end());
  if (cands.empty()) {
    throw std::runtime_error("no cublasLt algorithm could be timed precisely");
  }
  if (model_default.precise_us <= 0.0f) {
    throw std::runtime_error("the model's own cublasLt algorithm could not be timed precisely");
  }
  std::sort(
      cands.begin(), cands.end(), [](Candidate const& a, Candidate const& b) { return a.precise_us < b.precise_us; });

  std::printf("%-8s %-10s  %s\n", "us", "TFLOPs", "algorithm");
  for (auto const& c : cands) {
    std::printf("%-8.2f %-10.1f  %s\n", c.precise_us, Tflops(s, c.precise_us), c.label.c_str());
  }
  std::printf(
      "%-8.2f %-10.1f  %s\n",
      model_default.precise_us,
      Tflops(s, model_default.precise_us),
      model_default.label.c_str());

  TuneResult out;
  // The model's pick competes too: the quick pass that chose the refined set can miss it.
  out.model_default_us = model_default.precise_us;
  out.model_default_wins = model_default.precise_us > 0.0f && model_default.precise_us <= cands.front().precise_us;
  out.best = out.model_default_wins ? model_default : cands.front();
  out.gemm_ex_us = p.PreciseTime([&] { return p.LaunchGemmEx(cublas); });
  std::printf("%-8.2f %-10.1f  cublasGemmEx DEFAULT\n", out.gemm_ex_us, Tflops(s, out.gemm_ex_us));
  std::printf(
      "best cublasLt %.2f us vs the model's pick %.2f us (%.2fx) and cublasGemmEx %.2f us (%.2fx)\n",
      out.best.precise_us,
      out.model_default_us,
      out.model_default_us / out.best.precise_us,
      out.gemm_ex_us,
      out.gemm_ex_us / out.best.precise_us);
  return out;
}

void Usage() {
  std::printf(
      "Usage: nvLstmSingleStepGemmTune --model=lstm_s|lstm_m|lstm_l | --units=U --steps=T\n"
      "  [--candidates=N] [--exhaustive] [--reps=N] [--refine=N] [--max-candidates=N] [--workspace-mb=N]\n"
      "  [--device=N] [--cache=FILE] [--min-gain=PCT] [--sms=N] [--batch=B] [--k=K]\n"
      "Tunes the cublasLt algorithm of the half-precision precompute GEMM.  With --cache=FILE, also writes the\n"
      "result to FILE, which nvLstmInf reads with --lt-algo-cache=FILE.  An existing FILE is updated in place and\n"
      "other entries are kept.  An entry is written only when the tuned algorithm is at least --min-gain percent\n"
      "(default 2) faster than the model's own pick (the heuristic's first result without workspace); otherwise it\n"
      "is removed: a smaller difference is within the timing noise.  With --sms=N the entry is for a green\n"
      "context of N SMs (as nvLstmInf reports for its precompute).  --refine=N (N >= 1) re-times the best N\n"
      "candidates of the quick pass, as well as every heuristic candidate.\n");
}

Options Parse(int argc, char** argv) {
  Options opt;
  static option const long_opts[] = {
      {"model", required_argument, nullptr, 'm'},
      {"units", required_argument, nullptr, 'u'},
      {"steps", required_argument, nullptr, 's'},
      {"candidates", required_argument, nullptr, 'c'},
      {"exhaustive", no_argument, nullptr, 'e'},
      {"reps", required_argument, nullptr, 'r'},
      {"refine", required_argument, nullptr, 'R'},
      {"max-candidates", required_argument, nullptr, 'x'},
      {"workspace-mb", required_argument, nullptr, 'w'},
      {"device", required_argument, nullptr, 'd'},
      {"cache", required_argument, nullptr, 'o'},
      {"sms", required_argument, nullptr, 'S'},
      {"batch", required_argument, nullptr, 'b'},
      {"k", required_argument, nullptr, 'K'},
      {"min-gain", required_argument, nullptr, 'g'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};
  int c;
  while ((c = getopt_long(argc, argv, "", long_opts, nullptr)) != -1) {
    switch (c) {
      case 'm': {
        bool found = false;
        for (auto const& pr : PRESETS) {
          if (std::strcmp(pr.name, optarg) == 0) {
            opt.steps = pr.steps;
            opt.units = pr.units;
            found = true;
          }
        }
        if (!found) {
          throw std::runtime_error(std::string("unknown model ") + optarg);
        }
        break;
      }
      case 'u':
        opt.units = std::atoi(optarg);
        break;
      case 's':
        opt.steps = std::atoi(optarg);
        break;
      case 'c':
        opt.candidates = std::atoi(optarg);
        break;
      case 'e':
        opt.exhaustive = true;
        break;
      case 'r':
        opt.reps = std::atoi(optarg);
        break;
      case 'R':
        opt.refine = std::atoi(optarg);
        break;
      case 'x':
        opt.max_candidates = std::atoi(optarg);
        break;
      case 'w':
        opt.workspace_mb = static_cast<std::size_t>(std::atoi(optarg));
        break;
      case 'd':
        opt.device = std::atoi(optarg);
        break;
      case 'o':
        opt.cache_path = optarg;
        break;
      case 'S':
        opt.sms = std::atoi(optarg);
        break;
      case 'b':
        opt.batch = std::max(1, std::atoi(optarg));
        break;
      case 'K':
        opt.k = std::atoi(optarg);
        break;
      case 'g': {
        // the whole argument a finite number (not "nan", "inf" or "2x"); negative values fail the check below
        char* end = nullptr;
        opt.min_gain = std::strtod(optarg, &end);
        if (end == optarg || *end != '\0' || !std::isfinite(opt.min_gain)) {
          Usage();
          std::exit(2);
        }
        break;
      }
      default:
        Usage();
        std::exit(c == 'h' ? 0 : 2);
    }
  }
  if (opt.units <= 0 || opt.steps <= 0 || opt.refine <= 0 || opt.min_gain < 0.0) {
    Usage();
    std::exit(2);
  }
  if (!opt.cache_path.empty() && opt.batch > 1) {
    throw std::runtime_error("--cache: the cache has entries for one GEMM per call only");
  }
  return opt;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options const opt = Parse(argc, argv);
    CudaCheck(cudaSetDevice(opt.device));
    cudaDeviceProp prop{};
    CudaCheck(cudaGetDeviceProperties(&prop, opt.device));
    std::printf(
        "device: %s (%d SMs), cublasLt %zu, units=%d steps=%d\n",
        prop.name,
        prop.multiProcessorCount,
        cublasLtGetVersion(),
        opt.units,
        opt.steps);

    cublasLtHandle_t lt;
    cublasHandle_t cublas;
    cudaStream_t stream;
    CublasCheck(cublasLtCreate(&lt));
    CublasCheck(cublasCreate(&cublas));
    CUgreenCtx green_ctx = nullptr;
    int stream_sms = 0;  // the green context's SMs, 0 for the whole device
    if (opt.sms > 0 && opt.sms < prop.multiProcessorCount) {
      // A green context of the SMs left after splitting off the others, as nvLstmInf's precompute gets the SMs its
      // inference partitions leave; split at the finest granularity (no co-scheduling), so that it is opt.sms SMs
      // whenever the device allows a partition of that size.
      CudaCheck(cudaFree(nullptr));  // the primary context, current
      CUdevice dev;
      CUdevResource all, others, part;
      unsigned groups = 1;
      auto const others_count = static_cast<unsigned>(prop.multiProcessorCount - opt.sms);
      CuCheck(cuDeviceGet(&dev, opt.device));
      CuCheck(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM));
      CuCheck(cuDevSmResourceSplitByCount(
          &others, &groups, &all, &part, CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, others_count));
      if (groups != 1 || part.sm.smCount == 0) {
        throw std::runtime_error("cannot leave " + std::to_string(opt.sms) + " SMs");
      }
      CUdevResourceDesc desc;
      CUstream green_stream;
      CuCheck(cuDevResourceGenerateDesc(&desc, &part, 1));
      CuCheck(cuGreenCtxCreate(&green_ctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
      CuCheck(cuGreenCtxStreamCreate(&green_stream, green_ctx, CU_STREAM_NON_BLOCKING, 0));
      stream = green_stream;
      stream_sms = static_cast<int>(part.sm.smCount);
      std::printf("green context: %u SMs (asked for %d)\n", part.sm.smCount, opt.sms);
    } else {
      CudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    if (opt.batch > 1) std::printf("strided-batched: %d GEMMs per call\n", opt.batch);
    CublasCheck(cublasSetStream(cublas, stream));

    Shape const shape{GATES * opt.units, opt.steps, opt.k > 0 ? opt.k : opt.units};
    TuneResult result;
    {
      Problem p(shape, opt, lt, stream);
      result = Tune(p, opt, lt, cublas);
    }

    if (!opt.cache_path.empty()) {
      auto cache = lc::Load(opt.cache_path);
      lc::Entry e;
      e.key = lc::MakeKey(opt.device, stream_sms, DTYPE, COMPUTE, shape.m, shape.n, shape.k);
      e.workspace = result.best.workspace;
      e.algo = result.best.algo;
      // Without an entry the model keeps its own pick, so an entry is only worth writing if it is clearly faster.
      if (!result.model_default_wins &&
          result.best.precise_us * (1.0 + opt.min_gain / 100.0) <= result.model_default_us) {
        lc::Upsert(cache, e);
      } else {
        lc::Erase(cache, e.key);
        std::printf(
            "no cache entry: the best cublasLt algorithm is less than %.1f %% faster than the model's own pick\n",
            opt.min_gain);
      }
      if (!lc::Save(opt.cache_path, cache)) {
        throw std::runtime_error("cannot write " + opt.cache_path);
      }
      std::printf("cache written: %s (%zu entries)\n", opt.cache_path.c_str(), cache.size());
    }
    cudaStreamDestroy(stream);
    if (green_ctx != nullptr) cuGreenCtxDestroy(green_ctx);
    cublasDestroy(cublas);
    cublasLtDestroy(lt);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

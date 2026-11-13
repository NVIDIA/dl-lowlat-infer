/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// On-disk cache of tuned cublasLt algorithms, written by nvLstmSingleStepGemmTune and read by ModelSingleStep
// (nvLstmInf --lt-algo-cache=FILE).
//
// One text line per entry:
//   <device> <sm_count> <lt_version> <dtype> <compute> <m> <n> <k> <workspace> <w0> ... <w7>
// where <device> is the CUDA device name with spaces replaced by '_', <sm_count> the SMs the GEMM runs on (the
// device's, or its green context's when it runs on a part of the device), <dtype>/<compute> are the cudaDataType_t
// and cublasComputeType_t values, and w0..w7 are the opaque words of cublasLtMatmulAlgo_t in hex.  Lines starting with
// '#' are comments.
//
// The key deliberately excludes leading dimensions: the model's pitches differ from the tuner's.  A reader must run
// cublasLtMatmulAlgoCheck against its own descriptors and keep its heuristic pick when the check fails.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace nvidia::quant::lowlat_infer::lt_cache {

struct Key {
  std::string device;
  int sm_count = 0;
  std::size_t lt_version = 0;
  int dtype = 0;
  int compute = 0;
  int m = 0;
  int n = 0;
  int k = 0;

  bool operator==(Key const& o) const {
    return device == o.device && sm_count == o.sm_count && lt_version == o.lt_version && dtype == o.dtype &&
           compute == o.compute && m == o.m && n == o.n && k == o.k;
  }
};

struct Entry {
  Key key;
  std::size_t workspace = 0;
  cublasLtMatmulAlgo_t algo{};
};

/// sm_count: the SMs the GEMM runs on; <= 0 for the whole device.
inline Key MakeKey(int device_id, int sm_count, int dtype, int compute, int m, int n, int k) {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device_id);
  std::string name = prop.name;
  std::replace(name.begin(), name.end(), ' ', '_');
  return Key{name, sm_count > 0 ? sm_count : prop.multiProcessorCount, cublasLtGetVersion(), dtype, compute, m, n, k};
}

inline std::vector<Entry> Load(std::string const& path) {
  std::vector<Entry> entries;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream ss(line);
    Entry e;
    ss >> e.key.device >> e.key.sm_count >> e.key.lt_version >> e.key.dtype >> e.key.compute >> e.key.m >> e.key.n >>
        e.key.k >> e.workspace;
    for (auto& w : e.algo.data) {
      ss >> std::hex >> w >> std::dec;
    }
    if (!ss.fail()) {
      entries.push_back(e);
    }
  }
  return entries;
}

// Removes the entry with this exact key, if any.
inline void Erase(std::vector<Entry>& entries, Key const& key) {
  entries.erase(
      std::remove_if(entries.begin(), entries.end(), [&](Entry const& e) { return e.key == key; }), entries.end());
}

// Replaces an entry with the same key, otherwise appends.
inline void Upsert(std::vector<Entry>& entries, Entry const& e) {
  for (auto& x : entries) {
    if (x.key == e.key) {
      x = e;
      return;
    }
  }
  entries.push_back(e);
}

inline bool Save(std::string const& path, std::vector<Entry> const& entries) {
  std::ofstream out(path);
  if (!out) {
    return false;
  }
  out << "# nvLstmSingleStepGemmTune cache: device sm_count lt_version dtype compute m n k workspace algo[8]\n";
  for (auto const& e : entries) {
    out << e.key.device << ' ' << e.key.sm_count << ' ' << e.key.lt_version << ' ' << e.key.dtype << ' '
        << e.key.compute << ' ' << e.key.m << ' ' << e.key.n << ' ' << e.key.k << ' ' << e.workspace;
    for (auto w : e.algo.data) {
      out << ' ' << std::hex << w << std::dec;
    }
    out << '\n';
  }
  out.close();  // flushes: a failed final write shows only here
  return !out.fail();
}

inline Entry const* Find(std::vector<Entry> const& entries, Key const& key) {
  for (auto const& e : entries) {
    if (e.key == key) {
      return &e;
    }
  }
  return nullptr;
}

}  // namespace nvidia::quant::lowlat_infer::lt_cache

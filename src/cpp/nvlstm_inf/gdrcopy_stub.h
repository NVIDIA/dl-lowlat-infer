/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "util.h"

namespace gdrcopy_utils {

template <typename T>
class GdrcopyBuffer {
 public:
  explicit GdrcopyBuffer(std::size_t) {}
  T *GetDevicePtr() const { return nullptr; }
};

template <typename T>
using GdrcopyBufferPtr = std::unique_ptr<GdrcopyBuffer<T>>;

}  // namespace gdrcopy_utils

namespace nvidia::quant::lowlat_infer::ssgemv::cpu_pingpong_handles {

template <typename datatype_t>
inline void signal_release(gdrcopy_utils::GdrcopyBufferPtr<datatype_t> const &, datatype_t) {
  StateCheck(false);
}

template <typename datatype_t>
inline void signal_release(gdrcopy_utils::GdrcopyBufferPtr<datatype_t> const &, datatype_t, uint32_t *) {
  StateCheck(false);
}

}  // namespace nvidia::quant::lowlat_infer::ssgemv::cpu_pingpong_handles

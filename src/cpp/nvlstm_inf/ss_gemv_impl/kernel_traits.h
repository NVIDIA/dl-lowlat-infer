/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace nvidia::quant::lowlat_infer::ssgemv {

/**
 * Structure describing architecture properties
 */
struct Arch final {
  /// Maximum size of shared memory
  int MAX_SMEM_SIZE;
  /// Number of warps per block for medium size network running in TBC
  int TBC_LSTM_M_NUM_WARPS_PER_BLOCK;
  /// Warp group size for medium size network running in TBC
  int TBC_LSTM_M_WARP_GROUP_SIZE;
  /// Number of warps per block for medium size network running in device
  int DEV_LSTM_M_NUM_WARPS_PER_BLOCK;
  /// Warp group size for medium size network running in device
  int DEV_LSTM_M_WARP_GROUP_SIZE;
  /// Number of warps per block for large size network running in device
  int DEV_LSTM_L_NUM_WARPS_PER_BLOCK;
  /// Warp group size for large size network running in device
  int DEV_LSTM_L_WARP_GROUP_SIZE;
};

inline constexpr Arch SM_HEAVY_ARCH{
    .MAX_SMEM_SIZE = 227 * 1024,
    .TBC_LSTM_M_NUM_WARPS_PER_BLOCK = 8,
    .TBC_LSTM_M_WARP_GROUP_SIZE = 16,
    .DEV_LSTM_M_NUM_WARPS_PER_BLOCK = 8,
    .DEV_LSTM_M_WARP_GROUP_SIZE = 16,
    .DEV_LSTM_L_NUM_WARPS_PER_BLOCK = 16,
    .DEV_LSTM_L_WARP_GROUP_SIZE = 8,
};

inline constexpr Arch SM_LIGHT_ARCH{
    .MAX_SMEM_SIZE = 99 * 1024,
    .TBC_LSTM_M_NUM_WARPS_PER_BLOCK = 8,
    .TBC_LSTM_M_WARP_GROUP_SIZE = 32,
    .DEV_LSTM_M_NUM_WARPS_PER_BLOCK = 8,
    .DEV_LSTM_M_WARP_GROUP_SIZE = 16,
    .DEV_LSTM_L_NUM_WARPS_PER_BLOCK = 16,
    .DEV_LSTM_L_WARP_GROUP_SIZE = 4,
};

inline bool is_sm_heavy_arch(int sm_arch) { return sm_arch >= 900 && sm_arch < 1200; }

////////////////////////////////////////////////////////////////////////////////////////////////////

enum Data_kind { DATA_TYPE_FP16, DATA_TYPE_BF16, DATA_TYPE_FP32 };

template <Data_kind DATA_KIND>
struct Data_type {
  using Elem4_type = float4;
  using Elem2_type = float2;
  using Elem_type = float;
  static constexpr float4 zero4{0.f, 0.f, 0.f, 0.f};
  static constexpr float zero{0.f};
};

template <>
struct Data_type<DATA_TYPE_FP16> {
  using Elem4_type = uint2;
  using Elem2_type = uint32_t;
  using Elem_type  = uint16_t;
  static constexpr uint2 zero4{0, 0};
  static constexpr uint16_t zero{0};
};

template <>
struct Data_type<DATA_TYPE_BF16> {
  using Elem4_type = uint2;
  using Elem2_type = uint32_t;
  using Elem_type  = uint16_t;
  static constexpr uint2 zero4{0, 0};
  static constexpr uint16_t zero{0};
};


template <Arch ARCH_, Data_kind DATA_KIND_, bool STAGE_, bool DEVICE_ALWAYS_SPIN_, bool GDRCOPY_ENABLED_>
struct KernelDispatchTraits {
    static constexpr Data_kind DATA_KIND = DATA_KIND_;
    static constexpr bool STAGE = STAGE_;
    static constexpr bool DEVICE_ALWAYS_SPIN = DEVICE_ALWAYS_SPIN_;
    static constexpr bool GDRCOPY_ENABLED = GDRCOPY_ENABLED_;
    static constexpr Arch ARCH = ARCH_;
};

}  // namespace nvidia::quant::lowlat_infer::ssgemv
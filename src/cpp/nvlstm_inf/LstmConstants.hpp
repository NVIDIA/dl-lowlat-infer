/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <array>
#include <cstddef>

namespace nvidia::quant::lowlat_infer {

/**
 * Gates of LSTM used in import of LSTM data
 */
inline constexpr std::array kGates{'i', 'f', 'c', 'o'};

/**
 * Number of gates
 */
inline constexpr std::size_t kNumGates = kGates.size();

}  // namespace nvidia::quant::lowlat_infer
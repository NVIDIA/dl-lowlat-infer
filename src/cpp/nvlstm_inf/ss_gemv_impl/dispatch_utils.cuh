/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

 #pragma once

 #include "kernel_traits.h"

namespace nvidia::quant::lowlat_infer::ssgemv {
namespace kernel_dispatch {

// ============================================================================
// Layer 3: Dispatch 3 boolean parameters (STAGE, DEVICE_ALWAYS_SPIN, GDRCOPY_ENABLED)
// ============================================================================
template<typename Launcher>
void dispatch_3bools(bool s, bool p, bool g, Launcher&& launcher) {
    if      (!s && !p && !g) launcher.template operator()<false, false, false>();
    else if (!s && !p &&  g) launcher.template operator()<false, false, true>();
    else if (!s &&  p && !g) launcher.template operator()<false, true,  false>();
    else if (!s &&  p &&  g) launcher.template operator()<false, true,  true>();
    else if ( s && !p && !g) launcher.template operator()<true,  false, false>();
    else if ( s && !p &&  g) launcher.template operator()<true,  false, true>();
    else if ( s &&  p && !g) launcher.template operator()<true,  true,  false>();
    else                     launcher.template operator()<true,  true,  true>();
}

// ============================================================================
// Layer 2: Dispatch data type then delegate to dispatch_3bools
// ============================================================================
template<Arch A, bool EnableFP32 = true, typename KernelLauncher>
void dispatch_4d(Data_kind dtype, bool s, bool p, bool g, KernelLauncher&& launcher) {
    if constexpr (EnableFP32) {
        if (dtype == DATA_TYPE_FP32) {
            dispatch_3bools(s, p, g, [&]<bool S, bool P, bool G>() {
                launcher.template operator()<A, DATA_TYPE_FP32, S, P, G>();
            });
            return;
        }
    }
    if (dtype == DATA_TYPE_FP16) {
        dispatch_3bools(s, p, g, [&]<bool S, bool P, bool G>() {
            launcher.template operator()<A, DATA_TYPE_FP16, S, P, G>();
        });
    }
    else if (dtype == DATA_TYPE_BF16) {
        dispatch_3bools(s, p, g, [&]<bool S, bool P, bool G>() {
            launcher.template operator()<A, DATA_TYPE_BF16, S, P, G>();
        });
    }
    else {
        // Should never reach here
        fprintf(stderr, "Error: Unsupported dtype=%d\n", dtype);
    }
}

// ============================================================================
// Layer 1: Top-level dispatcher - dispatches data type and architecture
// ============================================================================
template<bool EnableFP32 = true, bool ArchDispatch = false, typename KernelLauncher>
void dispatch_kernel_variants(
    Data_kind dtype,
    bool stage,
    bool device_always_spin,
    bool gdrcopy_enabled,
    int sm_arch,
    KernelLauncher&& launcher) {
    if constexpr (ArchDispatch) {
        bool is_heavy = is_sm_heavy_arch(sm_arch);
        if (is_heavy) {
            dispatch_4d<SM_HEAVY_ARCH, EnableFP32>(
                dtype, stage, device_always_spin, gdrcopy_enabled, launcher);
        } else {
            dispatch_4d<SM_LIGHT_ARCH, EnableFP32>(
                dtype, stage, device_always_spin, gdrcopy_enabled, launcher);
        }
    } else {
        // When ArchDispatch=false (default), sm_arch is ignored and always use SM_HEAVY_ARCH
        dispatch_4d<SM_HEAVY_ARCH, EnableFP32>(
            dtype, stage, device_always_spin, gdrcopy_enabled, launcher);
    }
}
    
} // namespace kernel_dispatch
} // namespace nvidia::quant::lowlat_infer
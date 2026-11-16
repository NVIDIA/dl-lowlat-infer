/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Tagged words for exchanging data between CTAs (README.md).
//
// A tagged word is 8 bytes: a 32-bit payload and a 32-bit epoch, written with one relaxed store. An aligned 8-byte
// access is single-copy atomic, so a reader that sees the epoch it waits for also holds that store's payload: no fence
// and no separate flag. A tagged word orders only its own 8 bytes; data elsewhere needs acquire / release.
//
// Words are reused by the parity of the epoch, two slots per word, and zeroed before use: a reader tests the epoch for
// equality only, so a zeroed slot must never match (epochs start at 1).

#include <cuda/atomic>
#include <stdint.h>

namespace nvidia::quant::lowlat_infer::ssgemv {

/// The 8-byte word of a 32-bit payload and its epoch (in the high half).
__device__ __forceinline__ unsigned long long tagged_word(unsigned payload, unsigned epoch) {
  return (static_cast<unsigned long long>(epoch) << 32) | payload;
}

/// Stores the word to global memory, visible to every CTA of the device (relaxed: orders only its own 8 bytes).
__device__ __forceinline__ void tagged_store(unsigned long long* p, unsigned long long word) {
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(*p).store(word, cuda::memory_order_relaxed);
}
/// Loads a word that other CTAs store (relaxed).
__device__ __forceinline__ unsigned long long tagged_load(unsigned long long* p) {
  return cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(*p).load(cuda::memory_order_relaxed);
}
/// The same with acquire semantics: also orders the reads that follow after the stores that preceded the word's store
/// (a signal for data elsewhere).
__device__ __forceinline__ unsigned long long tagged_load_acquire(unsigned long long* p) {
  return cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(*p).load(cuda::memory_order_acquire);
}

/// The same with release semantics: the stores that preceded it, and the writes this thread has seen, are visible to a
/// reader that sees the word with tagged_load_acquire.
__device__ __forceinline__ void tagged_store_release(unsigned long long* p, unsigned long long word) {
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(*p).store(word, cuda::memory_order_release);
}

/// Spins until the word carries the epoch; returns its payload. load: tagged_load (the default) or tagged_load_acquire.
template <typename LOAD = decltype(&tagged_load)>
__device__ __forceinline__ unsigned tagged_poll(unsigned long long* p, unsigned epoch, LOAD load = tagged_load) {
  unsigned long long w;
  do {
    w = load(p);
  } while (static_cast<unsigned>(w >> 32) != epoch);
  return static_cast<unsigned>(w);
}

/// Spins until the word carries `epoch` or a later one (epochs compared modulo 2^32); returns the word. For a word
/// whose writer may move on to later epochs before every reader has seen the one it waits for.
template <typename LOAD = decltype(&tagged_load)>
__device__ __forceinline__ unsigned long long tagged_poll_at_least(
    unsigned long long* p, unsigned epoch, LOAD load = tagged_load) {
  unsigned long long w;
  do {
    w = load(p);
  } while (static_cast<int>(static_cast<unsigned>(w >> 32) - epoch) < 0);
  return w;
}

}  // namespace nvidia::quant::lowlat_infer::ssgemv

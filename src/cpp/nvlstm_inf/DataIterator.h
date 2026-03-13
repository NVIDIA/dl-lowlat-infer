/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "util.h"

namespace nvidia::quant::lowlat_infer {

template <typename TVector>
class DataIterator {
 public:
  using VectorType = TVector;
  using value_type = typename VectorType::value_type;

  DataIterator(
      std::shared_ptr<VectorType const> data,
      std::size_t num_samples_total_,
      std::uint32_t num_features_,
      std::unique_ptr<util::DataSampler> seq,
      value_type const* data_ptr = nullptr)
      : data_{std::move(data)},
        data_ptr_{data_ptr == nullptr ? data_->data() : data_ptr},
        num_samples_total_{num_samples_total_},
        num_features_{num_features_},
        seq_{std::move(seq)} {
    StateCheck(data_->size() == num_samples_total_ * num_features_);
  };

  template <typename TShape>
  DataIterator(
      util::Tensor<VectorType, TShape> const& data,  // NOLINT(cppcoreguidelines-pro-type-member-init)
      std::unique_ptr<util::DataSampler> seq,
      value_type const* data_ptr = nullptr)
      : DataIterator{
            data.DataShared(),
            data.Shape().at(0),
            static_cast<std::uint32_t>(data.Shape().at(1)),
            std::move(seq),
            data_ptr} {};

  DataIterator(DataIterator const& other) = delete;
  DataIterator& operator=(DataIterator const& other) = delete;

  value_type const* Next() {
    auto const sample_idx = seq_->Next();
    return data_ptr_ + sample_idx * num_features_;
  }

  void MemcpyNext(value_type* buf) {
    auto const addr = Next();
    std::memcpy(buf, addr, num_features_ * sizeof(value_type));
  }

  void Reset() { seq_->Reset(); }

  VectorType const& Data() const { return *data_; }

  [[nodiscard]] std::size_t NumSamples() const { return seq_->Size(); }
  [[nodiscard]] std::size_t NumExcessSamples() const { return num_samples_total_ - NumSamples(); }
  [[nodiscard]] auto NumFeatures() const { return num_features_; }

 private:
  std::shared_ptr<VectorType const> const data_;
  value_type const* const data_ptr_;
  std::size_t const num_samples_total_;
  std::uint32_t const num_features_;
  std::unique_ptr<util::DataSampler> seq_;
};

}  // namespace nvidia::quant::lowlat_infer

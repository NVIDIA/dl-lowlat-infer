/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>

#include <nvlstm_inf/ModelDef.h>


#ifndef TEST_RESOURCES_DIR
#error "Define TEST_RESOURCES_DIR"
#endif

using namespace nvidia::quant::lowlat_infer;

TEST(ModelDef, Load) {
  auto path = std::filesystem::path(TEST_RESOURCES_DIR) / "lstm_s";
  ASSERT_TRUE(std::filesystem::exists(path));
  ModelDef model_def{path.c_str()};
  EXPECT_EQ(model_def.NumSteps(), 64);
  EXPECT_EQ(model_def.NumFeatures(), 128);
  EXPECT_EQ(model_def.NumLayers(), 2);

  for (std::size_t level = 0; level < 2; ++level) {
    auto const &w = model_def.LstmW().at(level);
    if (level == 0) {
      EXPECT_EQ(w.Shape(), util::ShapeVector({1, 384, 128}));
    } else {
      EXPECT_EQ(w.Shape(), util::ShapeVector({1, 384, 96}));
    }
    auto const &r = model_def.LstmR().at(level);
    EXPECT_EQ(r.Shape(), util::ShapeVector({1, 384, 96}));
    auto const &b = model_def.LstmB().at(level);
    EXPECT_EQ(b.Shape(), util::ShapeVector({1, 768}));
  }

  EXPECT_EQ(model_def.MatMul().Shape().at(0), model_def.NumUnits());
  EXPECT_EQ(model_def.MatMul().Shape().at(1), 1);
  EXPECT_TRUE(model_def.HasBias());
  EXPECT_FLOAT_EQ(model_def.Bias(), 0.065197498);
  EXPECT_EQ(model_def.NumLstmWeights(), 384 * 128 + 384 * 96 + 2 * 384 * 96 + 2 * 768);

  EXPECT_EQ(model_def.OffsetW('i', true), 0);
  EXPECT_EQ(model_def.OffsetW('i', false), 0);
  EXPECT_EQ(model_def.OffsetW('o', true), 96 * 128);
  EXPECT_EQ(model_def.OffsetW('o', false), 96 * 96);
  EXPECT_EQ(model_def.OffsetW('f', true), 192 * 128);
  EXPECT_EQ(model_def.OffsetW('f', false), 192 * 96);
  EXPECT_EQ(model_def.OffsetW('c', true), 288 * 128);
  EXPECT_EQ(model_def.OffsetW('c', false), 288 * 96);
  EXPECT_EQ(model_def.OffsetR('i'), 0);
  EXPECT_EQ(model_def.OffsetR('o'), 96 * 96);
  EXPECT_EQ(model_def.OffsetR('f'), 192 * 96);
  EXPECT_EQ(model_def.OffsetR('c'), 288 * 96);
  EXPECT_EQ(model_def.OffsetB('i', false), 0);
  EXPECT_EQ(model_def.OffsetB('i', true), 384);
  EXPECT_EQ(model_def.OffsetB('o', false), 96);
  EXPECT_EQ(model_def.OffsetB('o', true), 480);
  EXPECT_EQ(model_def.OffsetB('f', false), 192);
  EXPECT_EQ(model_def.OffsetB('f', true), 6*96);
  EXPECT_EQ(model_def.OffsetB('c', false), 288);
  EXPECT_EQ(model_def.OffsetB('c', true), 7*96);
}

TEST(ModelDef, Flops) {
  std::filesystem::path model_dir(TEST_RESOURCES_DIR);

  ModelDef model_A{(model_dir / "lstm_s").c_str()};
  EXPECT_EQ(model_A.Flops(2, 2), 20'717'760);
  ModelDef model_B{(model_dir / "lstm_m").c_str()};
  EXPECT_EQ(model_B.Flops(2, 2), 122'911'040);
  ModelDef model_C{(model_dir / "lstm_l").c_str()};
  EXPECT_EQ(model_C.Flops(2, 2), 4'277'044'672);
}

TEST(ModelDef, SerialSolNanos) {
  std::filesystem::path model_dir(TEST_RESOURCES_DIR);
  
  auto constexpr gemm = 156000;
  auto constexpr other = 19500;
  auto constexpr cost_activation = 2;
  ModelDef model_A{(model_dir / "lstm_s").c_str()};
  ModelDef model_B{(model_dir / "lstm_m").c_str()};
  ModelDef model_C{(model_dir / "lstm_l").c_str()};
  EXPECT_EQ(static_cast<int>(model_A.SerialSolNanos(gemm, other, cost_activation, cost_activation)), 565);
  EXPECT_EQ(static_cast<int>(model_B.SerialSolNanos(gemm, other, cost_activation, cost_activation)), 3470);
  EXPECT_EQ(static_cast<int>(model_C.SerialSolNanos(gemm, other, cost_activation, cost_activation)), 127272);

  EXPECT_EQ(static_cast<int>(model_A.FlopsPerNanoSol(gemm, other, cost_activation, cost_activation)), 36611);
  EXPECT_EQ(static_cast<int>(model_B.FlopsPerNanoSol(gemm, other, cost_activation, cost_activation)), 35415);
  EXPECT_EQ(static_cast<int>(model_C.FlopsPerNanoSol(gemm, other, cost_activation, cost_activation)), 33605);
}

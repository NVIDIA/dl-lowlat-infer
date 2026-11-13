/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "nvlstm_inf/RunDefaults.h"

using namespace nvidia::quant::lowlat_infer::run_defaults;

TEST(RunDefaultsTest, ParseChoice) {
  EXPECT_EQ(ParseChoice(nullptr, "opt"), Choice::kOn);
  EXPECT_EQ(ParseChoice("1", "opt"), Choice::kOn);
  EXPECT_EQ(ParseChoice("0", "opt"), Choice::kOff);
  EXPECT_THROW(ParseChoice("2", "opt"), std::invalid_argument);
  EXPECT_THROW(ParseChoice("", "opt"), std::invalid_argument);
}

TEST(RunDefaultsTest, ResolveAutomatic) {
  EXPECT_TRUE(Resolve(Choice::kAuto, "opt", true, "X=ON", "").on);
  EXPECT_EQ(Resolve(Choice::kAuto, "opt", true, "X=ON", "").note, "default");
  auto const not_built = Resolve(Choice::kAuto, "opt", false, "X=ON", "");
  EXPECT_FALSE(not_built.on);
  EXPECT_EQ(not_built.note, "not built");
  auto const not_applicable = Resolve(Choice::kAuto, "opt", true, "X=ON", "needs lstm_l");
  EXPECT_FALSE(not_applicable.on);
  EXPECT_EQ(not_applicable.note, "off: needs lstm_l");
}

TEST(RunDefaultsTest, ResolveRequiredAndOff) {
  EXPECT_TRUE(Resolve(Choice::kOn, "opt", true, "X=ON", "").on);
  EXPECT_THROW(Resolve(Choice::kOn, "opt", false, "X=ON", ""), std::invalid_argument);
  EXPECT_THROW(Resolve(Choice::kOn, "opt", true, "X=ON", "needs lstm_l"), std::invalid_argument);
  auto const off = Resolve(Choice::kOff, "opt", true, "X=ON", "");
  EXPECT_FALSE(off.on);
  EXPECT_EQ(off.note, "--opt=0");
}

TEST(RunDefaultsTest, Refused) {
  auto const fallback = Refused(Choice::kAuto, "opt", "does not fit", "the other path runs");
  EXPECT_FALSE(fallback.on);
  EXPECT_EQ(fallback.note, "off: does not fit; the other path runs");
  EXPECT_THROW(Refused(Choice::kOn, "opt", "does not fit", "the other path runs"), std::invalid_argument);
}

TEST(RunDefaultsTest, ResolveGdrcopy) {
  int checks = 0;
  auto const works = [&checks] {
    ++checks;
    return GdrcopyCheck{true, "gdrdrv"};
  };
  auto const fails = [&checks] {
    ++checks;
    return GdrcopyCheck{false, "no module"};
  };

  auto const on = ResolveGdrcopy(Choice::kAuto, true, true, "", works);
  EXPECT_TRUE(on.on);
  EXPECT_EQ(on.note, "gdrdrv; default");
  auto const fallback = ResolveGdrcopy(Choice::kAuto, true, true, "", fails);
  EXPECT_FALSE(fallback.on);
  EXPECT_EQ(fallback.note, "off: no module");
  EXPECT_EQ(checks, 2);

  // not checked where it is not used by default
  EXPECT_FALSE(ResolveGdrcopy(Choice::kAuto, true, false, "", works).on);
  EXPECT_FALSE(ResolveGdrcopy(Choice::kAuto, true, true, "FP16 only", works).on);
  EXPECT_FALSE(ResolveGdrcopy(Choice::kAuto, false, true, "", works).on);
  EXPECT_FALSE(ResolveGdrcopy(Choice::kOff, true, true, "", works).on);
  EXPECT_EQ(checks, 2);

  // required: wherever it works, an error where it does not
  EXPECT_TRUE(ResolveGdrcopy(Choice::kOn, true, false, "FP16 only", works).on);
  EXPECT_THROW(ResolveGdrcopy(Choice::kOn, true, true, "", fails), std::runtime_error);
  EXPECT_THROW(ResolveGdrcopy(Choice::kOn, false, true, "", works), std::invalid_argument);
}

TEST(RunDefaultsTest, ResolveStepsPerGraph) {
  EXPECT_EQ(ResolveStepsPerGraph(std::nullopt, true), kDefaultStepsPerGraph);
  EXPECT_EQ(ResolveStepsPerGraph(std::nullopt, false), 1u);
  EXPECT_EQ(ResolveStepsPerGraph(4, true), 4u);
  EXPECT_EQ(ResolveStepsPerGraph(1, true), 1u);
  EXPECT_THROW(ResolveStepsPerGraph(3, true), std::invalid_argument);
  EXPECT_THROW(ResolveStepsPerGraph(0, true), std::invalid_argument);
  EXPECT_THROW(ResolveStepsPerGraph(-2, true), std::invalid_argument);
  EXPECT_THROW(ResolveStepsPerGraph(2, false), std::invalid_argument);
}

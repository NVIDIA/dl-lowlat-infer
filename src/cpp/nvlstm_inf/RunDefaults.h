/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// nvLstmInf's run-time paths that are on by default where they work: GDRCopy, the resident SingleStep precompute and
// the FP16 register lstm_l kernel in its precompute graph. Each option is unset (automatic), `=1` (required: an error
// where it cannot run) or `=0` (off); what it resolved to is printed with its source.

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>

namespace nvidia::quant::lowlat_infer::run_defaults {

enum class Choice { kAuto, kOn, kOff };

// `--option` (no value) or `--option=1`: on; `--option=0`: off.
inline Choice ParseChoice(char const *value, std::string const &option) {
  if (value == nullptr) {
    return Choice::kOn;
  }
  std::string const text{value};
  if (text == "1") {
    return Choice::kOn;
  }
  if (text == "0") {
    return Choice::kOff;
  }
  throw std::invalid_argument("--" + option + " takes 0 or 1, not '" + text + "'");
}

struct Setting {
  bool on = false;
  std::string note;  // where the value came from, or why it is off
};

// An option that runs where it is built and applicable. not_applicable: empty when it applies, otherwise why not.
inline Setting Resolve(
    Choice choice,
    std::string const &option,
    bool built,
    std::string const &build_hint,
    std::string const &not_applicable) {
  switch (choice) {
    case Choice::kOff:
      return {false, "--" + option + "=0"};
    case Choice::kOn:
      if (!built) {
        throw std::invalid_argument("--" + option + " requires a build with " + build_hint);
      }
      if (!not_applicable.empty()) {
        throw std::invalid_argument("--" + option + " " + not_applicable);
      }
      return {true, "--" + option};
    case Choice::kAuto:
    default:
      if (!built) {
        return {false, "not built"};
      }
      if (!not_applicable.empty()) {
        return {false, "off: " + not_applicable};
      }
      return {true, "default"};
  }
}

// After a check at run time refused an option that resolved to on (e.g. the resident precompute's fit check): an error
// if the option was required, otherwise off with the reason and what runs instead.
inline Setting Refused(
    Choice choice, std::string const &option, std::string const &reason, std::string const &instead) {
  if (choice == Choice::kOn) {
    throw std::invalid_argument("--" + option + ": " + reason);
  }
  return {false, "off: " + reason + "; " + instead};
}

inline constexpr unsigned kDefaultStepsPerGraph = 8;

struct GdrcopyCheck {
  bool ok = false;
  std::string description;  // the backend, or why GDRCopy does not work
};

// GDRCopy: automatic on x86 where it applies (not_applicable empty) and works here (check: opens GDRCopy and maps a
// device page; called only when GDRCopy may be used). Required (`=1`), it runs wherever it works.
inline Setting ResolveGdrcopy(
    Choice choice,
    bool built,
    bool x86,
    std::string const &not_applicable,
    std::function<GdrcopyCheck()> const &check) {
  if (choice == Choice::kOff) {
    return {false, "--gdrcopy=0"};
  }
  if (!built) {
    if (choice == Choice::kOn) {
      throw std::invalid_argument("--gdrcopy requires a build configured with -DUSE_GDRCOPY=ON");
    }
    return {false, "not built"};
  }
  if (choice == Choice::kAuto && !x86) {
    return {false, "off: on by default on x86 only"};
  }
  if (choice == Choice::kAuto && !not_applicable.empty()) {
    return {false, "off: " + not_applicable};
  }
  auto const result = check();
  if (!result.ok) {
    if (choice == Choice::kOn) {
      throw std::runtime_error("--gdrcopy: GDRCopy does not work here: " + result.description);
    }
    return {false, "off: " + result.description};
  }
  return {true, result.description + (choice == Choice::kOn ? "; --gdrcopy" : "; default")};
}

// Steps per graph of the in-graph lstm_l kernel: given (1 or even), or by default 8 (with GDRCopy the most throughput
// on the bare-metal RTX PRO 6000 and GH200 at no P99 cost; without it level with or ahead of 2).
inline unsigned ResolveStepsPerGraph(std::optional<int> given, bool in_graph) {
  if (given) {
    if (!in_graph) {
      throw std::invalid_argument("--steps-per-graph requires --lstm-l-in-graph");
    }
    if (*given < 1 || (*given > 1 && *given % 2 != 0)) {
      throw std::invalid_argument("--steps-per-graph must be 1 or even");
    }
    return static_cast<unsigned>(*given);
  }
  if (!in_graph) {
    return 1;
  }
  return kDefaultStepsPerGraph;
}

}  // namespace nvidia::quant::lowlat_infer::run_defaults

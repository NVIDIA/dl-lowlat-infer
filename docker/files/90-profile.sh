#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Check if running in privileged mode by testing if /proc/sys/kernel/perf_event_paranoid is writable
if [[ $(id -u) -eq 0 ]] || groups | grep -q '\bsudo\b'; then
  if sudo test -w /proc/sys/kernel/perf_event_paranoid; then
    sudo sh -c "echo 1 > /proc/sys/kernel/perf_event_paranoid"
  fi
  # nsys status -e
fi

#!/usr/bin/python3

# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref", type=str, required=True)
    parser.add_argument("-o", "--out", type=str, required=True)
    parser.add_argument("-n", type=int, help="Number of samples to check")
    parser.add_argument("-e", "--eps", type=float, default=5.0e-2, help="Precision to check for")
    parser.add_argument("-v", "--verbose", type=int, help="Be verbose for given number of steps")
    args = parser.parse_args()

    ref = np.load(args.ref)
    out = np.load(args.out)

    print(f"Shape of ref data is {ref.shape}")
    print(f"Shape of out data is {out.shape}")

    size = min(ref.shape[0], out.shape[0])
    size = size if args.n is None else min(size, args.n)

    print(f"Comparing {size} steps")

    diff = ref[0:size, 0] - out[0:size, 0]

    if args.verbose:
        n = min(args.verbose, size)
        for r, o in zip(ref[0:n, 0], out[0:n, 0]):
            print(
                f"Reference = {r}, Output = {o}, Difference = {abs(r - o)}, Result is { '💚' if abs(r - o) < args.eps else '💔'}"
            )
        print(f"Maximum absolute difference is {np.max(np.abs(diff))}")

    if np.max(np.abs(diff)) > args.eps:
        raise AssertionError("Test failed! 💔")
    else:
        print("Test Passed! 💚")

#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import pathlib
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--num-inputs", type=int, required=True)
    parser.add_argument("--num-units", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    num_steps = args.num_steps
    num_inputs = args.num_inputs
    num_units = args.num_units
    num_layers = args.num_layers

    model = args.model

    if args.seed is not None:
        np.random.seed(args.seed)

    output_dir = pathlib.Path(args.output_dir + "/" + model)
    print(f"Output directory: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    output_dir.mkdir(parents=True, exist_ok=False)

    for layer_index in range(num_layers):
        num_features = num_inputs if layer_index == 0 else num_units
        # Cutoff like in PyTorch: Xavier uniform initialization (https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py)
        cutoff = np.sqrt(3) * np.sqrt(2 / (num_units * 4 + num_features))
        input_W = np.random.uniform(-cutoff, cutoff, size=(1, num_units * 4, num_features)).astype(np.float32)
        np.save(str(output_dir / f"lstm_{layer_index}_W.npy"), input_W, allow_pickle=False)
        cutoff = np.sqrt(3) * np.sqrt(2 / (num_units * 4 + num_units))
        input_R = np.random.uniform(-cutoff, cutoff, size=(1, num_units * 4, num_units)).astype(np.float32)
        np.save(str(output_dir / f"lstm_{layer_index}_R.npy"), input_R, allow_pickle=False)
        # https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py
        cutoff = np.sqrt(1 / num_units)
        input_B = np.random.uniform(-cutoff, cutoff, size=(1, num_units * 8)).astype(np.float32)
        np.save(str(output_dir / f"lstm_{layer_index}_B.npy"), input_B, allow_pickle=False)

    cutoff = np.sqrt(1 / (num_units))
    mul = np.random.uniform(-cutoff, cutoff, size=(num_units, 1)).astype(np.float32)
    np.save(str(output_dir / "mul.npy"), mul, allow_pickle=False)
    add = np.random.uniform(-1, 1, size=(1,)).astype(np.float32)
    np.save(str(output_dir / "add.npy"), add, allow_pickle=False)

    np_input_size = np.array((num_steps, num_inputs), np.int32)
    np.save(str(output_dir / "input_size.npy"), np_input_size, allow_pickle=False)


if __name__ == "__main__":
    main()

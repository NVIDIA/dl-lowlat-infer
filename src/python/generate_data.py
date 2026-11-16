#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate uniformly distributed vectors and save them in NPY format.

For example: python src/python/generate_data.py --num-inputs 128 -o /tmp/lstm_m.npy.
"""

import argparse

import numpy as np


def generate_uniform_random_vectors(num_samples, num_inputs, cutoff=1.5, seed=None):
    """Generate num_samples random uniformly distributed vectors of size num_inputs.

    Args:
        num_samples (int): Number of vectors to generate
        num_inputs (int): Size of each vector
        cutoff (float): Range of uniform distribution [-cutoff, cutoff] (default: 1.5)
        seed (int, optional): Random seed for reproducibility

    Returns:
        numpy.ndarray: Array of shape (num_samples, num_inputs) containing the random vectors
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random vectors with uniform distribution between -cutoff and cutoff
    vectors = np.random.uniform(-cutoff, cutoff, size=(num_samples, num_inputs))

    return vectors.astype("float32")


def save_vectors(vectors, filename):
    """Save vectors to NPY format.

    Args:
        vectors (numpy.ndarray): Array of vectors to save
        filename (str): Output filename
    """
    np.save(filename, vectors, allow_pickle=False)
    print(f"Saved {vectors.shape[0]} vectors of size {vectors.shape[1]} to {filename}")


def main():
    NUM_SAMPLES = 1 << 20
    parser = argparse.ArgumentParser(description="Generate random uniformly distributed vectors and save in NPY format")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="Number of vectors to generate")
    parser.add_argument("--num-inputs", type=int, default=128, help="Size of each vector")
    parser.add_argument(
        "-o", "--output", type=str, default="random_vectors.npy", help="Output filename (default: random_vectors.npy)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Print additional information")
    parser.add_argument(
        "--cutoff", type=float, default=1.5, help="Range of uniform distribution [-cutoff, cutoff] (default: 1.5)"
    )
    args = parser.parse_args()

    if args.verbose:
        print(f"Generating {args.num_samples} random vectors of size {args.num_inputs}")
        print(f"Distribution: uniform between [-{args.cutoff}, {args.cutoff}]")
        if args.seed is not None:
            print(f"Using random seed: {args.seed}")

    # Generate vectors
    vectors = generate_uniform_random_vectors(
        num_samples=args.num_samples, num_inputs=args.num_inputs, cutoff=args.cutoff, seed=args.seed
    )

    # Save vectors
    save_vectors(vectors, args.output)

    if args.verbose:
        print(f"Array shape: {vectors.shape}")
        print(f"Array dtype: {vectors.dtype}")
        print(f"Memory usage: {vectors.nbytes / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()

#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit on error
set -u  # Exit on undefined variable

# Set GDRCOPY_TAG to default if not specified
# e.g. GDRCOPY_TAG=v2.6
GDRCOPY_TAG="${GDRCOPY_TAG:-v2.6}"

# Check if kernel module already exists
KERNEL_VERSION=$(uname -r)
DRV_DIR="mod-${KERNEL_VERSION}"
if [ -f "${DRV_DIR}/gdrdrv.ko" ]; then
    echo "Kernel module already exists at ${DRV_DIR}/gdrdrv.ko"
    echo "Skipping build."
    exit 0
fi

echo "Building gdrcopy with tag: ${GDRCOPY_TAG}"

GDRCOPY_DIR="gdrcopy-${GDRCOPY_TAG}"

# Clone the repository
if [ -d "${GDRCOPY_DIR}" ]; then
    echo "gdrcopy directory already exists, removing it..."
    rm -rf ${GDRCOPY_DIR}
fi

echo "Cloning gdrcopy repository..."
git clone --branch "${GDRCOPY_TAG}" --depth 1 https://github.com/NVIDIA/gdrcopy.git ${GDRCOPY_DIR}

# Build the driver
echo "Building gdrcopy driver..."
cd ${GDRCOPY_DIR}
make clean || { echo "ERROR: Failed to clean gdrcopy build"; exit 1; }
make driver || { echo "ERROR: Failed to build gdrcopy driver"; exit 1; }

# Go back to the original directory
cd ..

# Create directory for kernel module
echo "Creating directory for kernel version: ${KERNEL_VERSION}"
mkdir -p "${DRV_DIR}"

# Move the kernel module
echo "Moving kernel module to ${DRV_DIR}/"
mv ${GDRCOPY_DIR}/src/gdrdrv/gdrdrv.ko "${DRV_DIR}/"

# Clean up the gdrcopy repository
echo "Cleaning up gdrcopy repository..."
rm -rf ${GDRCOPY_DIR}

echo "Build completed successfully!"
echo "Kernel module location: ${DRV_DIR}/gdrdrv.ko"

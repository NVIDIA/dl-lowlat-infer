#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
 
set -e

THIS_DIR=$(dirname $0)

# Remove driver if loaded
if grep gdrdrv /proc/devices >/dev/null; then
    echo "Removing existing gdrdrv module..."
    set +e
    RMMOD_OUTPUT=$(sudo /sbin/rmmod gdrdrv 2>&1)
    RMMOD_RESULT=$?
    set -e
    
    if [ $RMMOD_RESULT -ne 0 ]; then
        echo "ERROR: Failed to remove existing gdrdrv kernel module"
        echo "Output: $RMMOD_OUTPUT"
        echo ""
        echo "Required docker run options:"
        echo "  --privileged"
        exit 1
    fi
    echo "Module removed successfully"
fi

# Insert driver  
KERNEL_VERSION=`uname -r`
DRV_DIR="mod-${KERNEL_VERSION}"
if [[ ! -d ${THIS_DIR}/${DRV_DIR} ]]; then
  echo "Directory $DRV_DIR does not exist! Please compile a kernel module for this kernel version."
  exit 1
else
  echo "Found $KERNEL_VERSION IN $DRV_DIR"
fi

TEMP_KO=$(mktemp /tmp/gdrdrv.ko.XXXXXX)
# Clean up on exit (even if script fails)
trap "rm -f $TEMP_KO" EXIT INT TERM
# Copy to secure temp location
cp "${THIS_DIR}/${DRV_DIR}/gdrdrv.ko" "$TEMP_KO"

# Try to load the module with detailed error handling
echo "Loading gdrdrv kernel module..."
set +e
INSMOD_OUTPUT=$(sudo /sbin/insmod $TEMP_KO dbg_enabled=0 info_enabled=1 2>&1)
INSMOD_RESULT=$?
set -e

if [ $INSMOD_RESULT -ne 0 ]; then
    echo "ERROR: Failed to load gdrdrv kernel module"
    echo "Output: $INSMOD_OUTPUT"
    echo ""
    echo "Required docker run options:"
    echo "  --privileged"
    exit 1
fi

echo "Module loaded successfully"

# Create device inodes
major=`fgrep gdrdrv /proc/devices | cut -b 1-4`
echo "INFO: driver major is $major"

# Remove old inodes just in case
if [ -e /dev/gdrdrv ]; then
    sudo rm /dev/gdrdrv
fi

echo "INFO: creating /dev/gdrdrv inode"
sudo mknod /dev/gdrdrv c $major 0
sudo chmod a+w+r /dev/gdrdrv

echo "gdrdrv setup complete"
#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CPP_BIN=$BASE_DIR/build/src/cpp/nvlstm_inf/nvLstmInf
PY_CODE=$BASE_DIR/test/python/streaming_ref_model.py
DATA_FILE=$BASE_DIR/test/resources/lstm_s.npy
LSTM_DIR=$BASE_DIR/test/resources/lstm_s

TS_OFFSET=${TS_OFFSET:-14236936057311868}

if [ ! -x $CPP_BIN ]; then
    echo "Cannot find executable $CPP_BIN"
    exit 1
fi

if [ ! -f $PY_CODE ]; then
    echo "Cannot find file $PY_CODE"
    exit 1
fi

REF_FILE=$(mktemp --suffix ".npy")
OUT_DIR=$(mktemp -d)

echo "Creating reference data in $REF_FILE"
python $PY_CODE --data-file $DATA_FILE --lstm-dir $LSTM_DIR --output $REF_FILE

echo "Creating data in $OUT_DIR"
$CPP_BIN --batch-size=1 --precision=32 --stage=1 --reap=1 --log-sec=0 --grace-period=0 \
    --use-omp=0 --algo=SingleStepStreaming --cpuset=0,1,2,3,4,5,6,7 \
    --sampler-type=Ring --ts-offset=${TS_OFFSET} lstm_s \
    $LSTM_DIR $DATA_FILE 1 $OUT_DIR

echo "Starting comparison ..."
echo python $BASE_DIR/src/python/check.py --ref $REF_FILE --out $OUT_DIR/lstm_s.1.1.npy --eps 0.0001 -v 10
python $BASE_DIR/src/python/check.py --ref $REF_FILE --out $OUT_DIR/lstm_s.1.1.npy --eps 0.0001 -v 10

# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import pathlib as _pl

import numpy as np
import onnx as _ox
import onnxruntime as _rt

parser = argparse.ArgumentParser(description="Run an ONNX LSTM inference smoke test")
parser.add_argument(
    "--model",
    type=_pl.Path,
    required=True,
    help="ONNX model, e.g. lstm_s.onnx written by src/python/create_onnx_model.py",
)
parser.add_argument("--data", type=_pl.Path, default=_pl.Path("test/resources/lstm_s.npy"))
args = parser.parse_args()

full_model_path = args.model

model = _ox.load_model(str(full_model_path))

# session = _rt.InferenceSession(str(full_model_path), providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
session = _rt.InferenceSession(str(full_model_path), providers=["CPUExecutionProvider"])

input_0 = session.get_inputs()[0]
input_name = input_0.name
print(input_name)

input_shape = input_0.shape
print(input_shape)

(_, num_steps, num_features) = input_shape

data = np.load(str(args.data))
print(data.shape)

assert num_features == data.shape[1]

window = data[:num_steps, :]
print(window.shape)

wdw = window.reshape([1, num_steps, num_features])
print(wdw.shape)

assert np.all(wdw[0, :, :] == window)


srun = session.run(None, {input_name: wdw})
print(srun)
inferred = session.run(None, {input_name: wdw})[0][0, 0]

offsets = range(256)

results = []
for offset in offsets:
    wdw = data[offset : num_steps + offset, :].reshape([1, num_steps, num_features])
    results.append(session.run(None, {input_name: wdw})[0][0, 0])

print([f"{x:.6f}" for x in results])

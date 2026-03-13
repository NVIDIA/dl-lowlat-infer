# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import onnx as _ox
import onnxruntime as _rt
import pathlib as _pl
import numpy as np


MODEL_PATH = _pl.Path("test/resources")
DATA_PATH = _pl.Path("test/resources")

model_version = 'A'
model_variant = ''

full_model_path = MODEL_PATH.joinpath(f"LSTM_{model_variant}{model_version}.onnx")

model = _ox.load_model(str(full_model_path))

#session = _rt.InferenceSession(str(full_model_path), providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
session = _rt.InferenceSession(str(full_model_path), providers=['CPUExecutionProvider'])

input_0 = session.get_inputs()[0]
input_name = input_0.name
print(input_name)

input_shape = input_0.shape
print(input_shape)

(_, num_steps, num_features) = input_shape

data = np.load(str(DATA_PATH.joinpath(f"LSTM_{model_version}.npy")))
print(data.shape)

assert num_features == data.shape[1]

window = data[:num_steps, :]
print(window.shape)

wdw = window.reshape([1, num_steps, num_features])
print(wdw.shape)

assert(np.all(wdw[0, :, :] == window))


srun = session.run(None, {input_name: wdw})
print(srun)
inferred = session.run(None, {input_name: wdw})[0][0, 0]

offsets = range(256)

results = []
for offset in offsets:
    wdw = data[offset:num_steps + offset, :].reshape([1, num_steps, num_features])
    results.append(session.run(None, {input_name: wdw})[0][0, 0])

print([f"{x:.6f}" for x in results]) 

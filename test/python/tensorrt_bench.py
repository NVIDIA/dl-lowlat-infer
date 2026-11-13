#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib as _pl
import time as _tm

import numpy as _np
import tensorrt as _trt
import torch as _tor

DEFAULT_NUM_INTERATIONS = 100

DEFAULT_DATA_PATH = "/data"
DEFAULT_MODELS_PATH = "/models"

device = _tor.device("cuda:0" if _tor.cuda.is_available() else "cpu")


def load_data(model: str, data_path: str = DEFAULT_DATA_PATH) -> _tor.Tensor:
    """Load data for a given network.

    Args:
        model: The network to load data for.
        data_path: The path to the data.

    Returns:
        The data for the given network.
    """
    path_model_data = _pl.Path(data_path) / f"LSTM_{model}.npy"
    assert path_model_data.exists(), f"Data for {model} not found at {path_model_data}"

    return _tor.from_numpy(_np.load(str(path_model_data))).to(device)


def time_inference(model: str, input: _tor.Tensor, num_iterations: int = DEFAULT_NUM_INTERATIONS) -> int:
    """Time inference for a given network.

    Args:
        model: The network to time inference for.
        input: The input to the network.
        num_iterations: The number of iterations to run inference for.

    Returns:
        The time it took to run inference for the given network.
    """
    path_model = _pl.Path(DEFAULT_MODELS_PATH) / f"LSTM_{model}.onnx"
    assert path_model.exists(), f"Model {model} not found at {path_model}"

    logger = _trt.Logger(_trt.Logger.INFO)
    builder = _trt.Builder(logger)
    network = builder.create_network(1 << int(_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = _trt.OnnxParser(network, logger)
    assert parser.parse_from_file(str(path_model)), "Failed to parse ONNX file"
    # input_tensor = network.get_input(0)
    # input_tensor.shape = (1, *input_tensor.shape[1:])
    config = builder.create_builder_config()
    config.clear_flag(_trt.BuilderFlag.TF32)
    serialized_engine = builder.build_serialized_network(network, config)
    runtime = _trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()
    output_tensor = network.get_output(0)
    output = _tor.ones(tuple(output_tensor.shape)).cuda()

    total = 0
    for i in range(num_iterations):
        buffers = [input[i:, :].data_ptr(), output.data_ptr()]
        tic = _tm.perf_counter_ns()
        assert context.execute_v2(buffers)
        toc = _tm.perf_counter_ns()
        total += toc - tic

    return total // num_iterations


if __name__ == "__main__":
    for model in ["A", "B", "C"]:
        input = load_data(model)
        print(f"Benchmarking LSTM_{model}: {time_inference(model, input) // 1000} us")
        del input

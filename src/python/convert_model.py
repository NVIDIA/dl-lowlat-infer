# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse as _ap
import pathlib
import shutil

import numpy as np
import onnx as _ox

# sys.path.append(os.getcwd())  # This is the Test Harness directory


def parse(args=None) -> _ap.Namespace:
    parser = _ap.ArgumentParser(description="ONNX Model Converter")
    parser.add_argument("model_path", help="model path")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument("model", help="model to run")
    return parser.parse_args(args)


def decode_tensor(t) -> np.ndarray:
    assert t.data_type == 1
    array = np.frombuffer(t.raw_data, np.dtype("<f4"))
    return array.reshape(t.dims)


# noinspection PyPep8Naming
def convert(model=None) -> None:
    if model is None:
        ns = parse()
        model_path = pathlib.Path(ns.model_path)
        output_dir = pathlib.Path(ns.output_dir)
        model = ns.model

    output_dir = output_dir / "models" / model
    if output_dir.exists():
        print(f"Warning: Deleting model output directory: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)

    output_dir.mkdir(parents=True, exist_ok=False)

    model_def = _ox.load_model(str(model_path))
    graph = model_def.graph

    assert len(graph.input) == 1
    input_dims = graph.input[0].type.tensor_type.shape.dim
    num_steps = input_dims[1].dim_value
    num_features = input_dims[2].dim_value
    print(f"Info:steps: {num_steps}, features: {num_features}")
    np_input_size = np.array((num_steps, num_features), np.int32)
    np.save(str(output_dir / "input_size.npy"), np_input_size, allow_pickle=False)
    initializers = {init.name: init for init in graph.initializer}
    expected_input_size = num_features
    layer_index = 0

    for node in graph.node:
        if node.op_type == "LSTM":
            input_W = decode_tensor(initializers[node.input[1]])
            np.save(str(output_dir / f"lstm_{layer_index}_W.npy"), input_W, allow_pickle=False)
            assert input_W.shape[2] == expected_input_size
            units = input_W.shape[1] // 4
            input_R = decode_tensor(initializers[node.input[2]])
            np.save(str(output_dir / f"lstm_{layer_index}_R.npy"), input_R, allow_pickle=False)
            assert input_R.shape == (1, units * 4, units)
            input_B = decode_tensor(initializers[node.input[3]])
            np.save(str(output_dir / f"lstm_{layer_index}_B.npy"), input_B, allow_pickle=False)
            assert input_B.shape == (1, units * 8)
            print(f"Info:LSTM_{layer_index} ({node.name}) - W: {input_W.shape}, R: {input_R.shape}, B: {input_B.shape}")
            expected_input_size = units
            layer_index += 1
        elif node.op_type == "Gemm":
            input_B = decode_tensor(initializers[node.input[1]])
            input_B = input_B.transpose(1, 0)
            np.save(str(output_dir / "mul.npy"), input_B, allow_pickle=False)
            print(f"Info: {node.name} - B: {input_B.shape}")
            expected_shape = (expected_input_size, 1) if layer_index > 0 else (expected_input_size * num_steps, 1)
            assert input_B.shape == expected_shape
            print(f"Info: {node.name} - B: {input_B.shape}")
            if len(node.input) > 2:
                input_B = decode_tensor(initializers[node.input[2]])
                np.save(str(output_dir / "add.npy"), input_B, allow_pickle=False)
                assert input_B.shape == (1,)
                print(f"Info: {node.name} - B: {input_B.shape}")
        elif node.op_type == "MatMul":
            input_B = decode_tensor(initializers[node.input[1]])
            np.save(str(output_dir / "mul.npy"), input_B, allow_pickle=False)
            expected_shape = (expected_input_size, 1) if layer_index > 0 else (expected_input_size * num_steps, 1)
            assert input_B.shape == expected_shape
            print(f"Info: {node.name} - B: {input_B.shape}")
        # elif node.op_type == "Add":
        #     input_B = decode_tensor(initializers[node.input[1]])
        #     np.save(str(output_dir / "add.npy"), input_B, allow_pickle=False)
        #     assert input_B.shape == (1,)
        #     print(f"Info: {node.name} - B: {input_B.shape}")

    print(f"Info:Model output directory: {output_dir}")
    for f in output_dir.iterdir():
        print(f"Info:{f}: {f.stat().st_size}")


if __name__ == "__main__":
    convert()

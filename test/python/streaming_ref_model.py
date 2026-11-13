# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import tqdm

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--lstm-dir",
        type=pathlib.Path,
        required=True,
        help="Directory with model data files for LSTM",
    )
    parser.add_argument(
        "-d", "--data-file", type=pathlib.Path, required=True, help="Filename of data being passed to model"
    )
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Filename for results")
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")
    args = parser.parse_args()
    assert args.lstm_dir.is_dir()
    assert args.data_file.is_file()
    return args


MAX_LAYERS = 20


def num_layers_(path: pathlib.Path) -> Optional[int]:
    nl = None
    for layer in range(MAX_LAYERS):
        if not (path / f"lstm_{layer}_W.npy").is_file():
            return nl
        nl = 1 if nl is None else nl + 1
    raise AssertionError("Exceeded max number of layers")


class LSTMStreaming(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, *, out_size=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = torch.nn.Linear(hidden_size, out_size, bias=True)

    def forward(self, x: torch.Tensor, st: Tuple[torch.Tensor, torch.Tensor]):
        r, rst = self.lstm(x, st)
        r = self.fc(r)
        return (r, rst)


def onnx_to_torch(w: torch.Tensor) -> torch.Tensor:
    """Convert tensor from ONNX order to torch order.

    Converts the tensor from iofc gate order (ONNX) to ifco gates order
    (torch).
    """
    i, o, f, c = torch.chunk(w, 4)
    return torch.concat((i, f, c, o))


def onnx_to_torch_bias(b: torch.Tensor) -> torch.Tensor:
    """Change gate order for bias."""
    return torch.concat(tuple(map(onnx_to_torch, torch.chunk(b, 2))))


@dataclass
class LSTMRun:
    lstm: LSTMStreaming
    h: torch.Tensor
    c: torch.Tensor

    def __call__(self, x):
        with torch.no_grad():
            out, (self.h, self.c) = self.lstm(x, (self.h, self.c))
            return out


def createLSTMRun(lstm_dir: pathlib.Path) -> LSTMRun:
    """Generate LSTMRun from files in data directory.

    Layout of data in the files is described in
    https://onnx.ai/onnx/operators/onnx__LSTM.html#lstm-7
    while the layout of the data for torch.nn.LSTM is described
    at https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """
    num_layers = num_layers_(lstm_dir)
    logger.debug(f"LSTM with {num_layers} layers")
    ws = [torch.tensor(np.load(lstm_dir / f"lstm_{layer}_W.npy")).squeeze() for layer in range(num_layers)]
    rs = [torch.tensor(np.load(lstm_dir / f"lstm_{layer}_R.npy")).squeeze() for layer in range(num_layers)]
    bs = [torch.tensor(np.load(lstm_dir / f"lstm_{layer}_B.npy")).squeeze() for layer in range(num_layers)]
    ws = list(map(onnx_to_torch, ws))
    rs = list(map(onnx_to_torch, rs))
    bs = list(map(onnx_to_torch_bias, bs))
    _, f_i = ws[0].shape
    _, f_h = rs[0].shape
    logger.debug(f"Input dimension {f_i} and feature dimension {f_h}")
    model = LSTMStreaming(f_i, f_h, num_layers)
    for layer in range(num_layers):
        getattr(model.lstm, f"weight_ih_l{layer}").data = ws[layer]
        getattr(model.lstm, f"weight_hh_l{layer}").data = rs[layer]
        b_ih, b_hh = torch.chunk(bs[layer], 2)
        getattr(model.lstm, f"bias_ih_l{layer}").data = b_ih
        getattr(model.lstm, f"bias_hh_l{layer}").data = b_hh
    model.fc.weight.data = torch.tensor(np.load(lstm_dir / "mul.npy")).squeeze()[torch.newaxis, :]
    model.fc.bias.data = torch.tensor(np.load(lstm_dir / "add.npy")).squeeze()

    return LSTMRun(model, torch.zeros(num_layers, f_h), torch.zeros(num_layers, f_h))


def main():
    args = parse_args()
    xs = torch.tensor(np.load(args.data_file))
    model = createLSTMRun(args.lstm_dir)
    res = np.zeros((xs.shape[0], 3), dtype=np.float32)
    for i, x in tqdm.tqdm(enumerate(xs), total=xs.shape[0]):
        x = x[torch.newaxis, :]
        start_ts = time.time_ns()
        r = model(x).item()
        end_ts = time.time_ns()
        if args.verbose:
            logger.info(f"{i} => {r} in {(end_ts - start_ts) / 1000}us")
        res[i, :] = np.array([r, start_ts, end_ts])
    if args.output is not None:
        np.save(str(args.output), res, allow_pickle=False)
        logger.info(f"Saved result to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    main()

#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch
import torch.nn as nn
import torch.onnx

lstm_s = {"input_size": 128, "hidden_size": 96, "num_layers": 2, "num_steps": 64}

lstm_m = {"input_size": 192, "hidden_size": 160, "num_layers": 3, "num_steps": 96}

lstm_l = {"input_size": 512, "hidden_size": 736, "num_layers": 4, "num_steps": 128}

MODELS = {"s": lstm_s, "m": lstm_m, "l": lstm_l}


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):  # hidden/cell states are omitted
        # h0/c0 will default to zeros if not provided
        out, (_, _) = self.lstm(x)
        out = self.fc(out[-1, :, :])
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys())
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir

    model_configs = {args.model: MODELS[args.model]} if args.model else MODELS

    torch.manual_seed(2132435)

    for name, model_config in model_configs.items():
        lstm_model = LSTMModel(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
        )
        lstm_model.eval()

        x = torch.randn(1, model_config["num_steps"], model_config["input_size"])

        torch.onnx.export(
            lstm_model,
            x,
            f"{output_dir}/lstm_{name}.onnx",
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            # No dynamic_shapes needed since batch is fixed
            # dynamo=True    # Use torch.export-based exporter if PyTorch is recent enough
        )

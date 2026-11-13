# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

install(PROGRAMS DESTINATION .)

install(FILES create_onnx_model.py convert_model.py generate_model.py generate_data.py check.py requirements.txt
        DESTINATION .)

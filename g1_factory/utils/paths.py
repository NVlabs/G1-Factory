# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asset paths for G1 Factory."""

import os
from pathlib import Path

_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = (_CHECKOUT_ROOT / Path(os.environ.get("G1_FACTORY_ROOT", ".")).expanduser()).resolve()


def asset_path(path: str | Path) -> str:
    """Return a Factory asset path."""
    return str((PROJECT_ROOT / "datasets" / "assets" / Path(path).expanduser()).resolve())

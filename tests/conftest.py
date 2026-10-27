# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from tiny_checkpoint import build_tiny_checkpoint

# Tests run on a tiny random checkpoint unless this names a real one, e.g.
# `google/t5gemma-2-270m-270m` (gated on Hugging Face).
REAL_MODEL = os.environ.get("T5GEMMA2_TEST_MODEL")

if not torch.cuda.is_available():
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    if REAL_MODEL:
        return REAL_MODEL
    path = str(tmp_path_factory.mktemp("tiny_t5gemma2"))
    build_tiny_checkpoint(path)
    return path

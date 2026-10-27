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

# Run the engine in the test process, so that all requests of a `generate`
# call are added before its first step. Otherwise they reach the engine one by
# one, and Model Runner V2 never reorders a batch (see test_text_only_lora.py).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    if REAL_MODEL:
        return REAL_MODEL
    path = str(tmp_path_factory.mktemp("tiny_t5gemma2"))
    build_tiny_checkpoint(path)
    return path

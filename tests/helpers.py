# SPDX-License-Identifier: Apache-2.0
"""HF reference generation and comparison helpers for the tests."""

import os

import numpy as np
import torch
from conftest import REAL_MODEL
from PIL import Image

MAX_TOKENS = 16
NUM_LOGPROBS = 5
# The tiny checkpoint runs in float32, where vLLM matches HF token for token.
# Set T5GEMMA2_TEST_DTYPE=float16 on GPUs without bfloat16 (e.g. T4).
DTYPE = getattr(
    torch,
    os.environ.get("T5GEMMA2_TEST_DTYPE", "bfloat16" if REAL_MODEL else "float32"),
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def random_image(seed: int, width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))


def llm_kwargs(**kwargs) -> dict:
    defaults = dict(dtype=str(DTYPE).removeprefix("torch."), max_model_len=1024, seed=0)
    if DEVICE == "cpu":
        defaults.update(enforce_eager=True, gpu_memory_utilization=0.5)
    return defaults | kwargs


def vllm_prompt(text: str, images: list, decoder_prompt: str | None) -> dict:
    encoder_prompt: dict = {"prompt": text}
    if images:
        encoder_prompt["multi_modal_data"] = {
            "image": images if len(images) > 1 else images[0]
        }
    if decoder_prompt is None:
        return encoder_prompt
    return {"encoder_prompt": encoder_prompt, "decoder_prompt": decoder_prompt}


@torch.inference_mode()
def hf_generate(model, processor, text, images, decoder_prompt):
    """Greedy HF generation; returns the new token IDs and their logprobs."""
    inputs = processor(text=text, images=images or None, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    kwargs = {}
    if decoder_prompt is not None:
        kwargs["decoder_input_ids"] = processor.tokenizer(
            decoder_prompt, return_tensors="pt"
        ).input_ids
    inputs = {k: v.to(DEVICE) for k, v in (inputs | kwargs).items()}
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        min_new_tokens=MAX_TOKENS,
        do_sample=False,
        output_logits=True,
        return_dict_in_generate=True,
    )
    num_prefix = 1 if decoder_prompt is None else kwargs["decoder_input_ids"].shape[1]
    logprobs = [torch.log_softmax(s[0].float(), dim=-1) for s in out.logits]
    return out.sequences[0, num_prefix:].tolist(), logprobs


def assert_matches_hf(hf_output, vllm_output, name: str) -> None:
    """vLLM must produce HF's tokens with the same logprobs.

    On a real checkpoint in bfloat16, the two may diverge at a near tie,
    where each token must be among the other's top logprobs.
    """
    hf_ids, hf_logprobs = hf_output
    vllm_ids = list(vllm_output.token_ids)
    for t, (hf_tok, vllm_tok) in enumerate(zip(hf_ids, vllm_ids)):
        vllm_top = vllm_output.logprobs[t]
        if hf_tok == vllm_tok and not REAL_MODEL:
            diff = abs(hf_logprobs[t][hf_tok].item() - vllm_top[hf_tok].logprob)
            assert diff < 1e-3, f"{name}: logprob of token {t} differs by {diff}"
        elif hf_tok != vllm_tok:
            assert REAL_MODEL, f"{name}: token {t} differs: {hf_ids} vs {vllm_ids}"
            hf_top = hf_logprobs[t].topk(NUM_LOGPROBS).indices.tolist()
            assert hf_tok in vllm_top and vllm_tok in hf_top, (
                f"{name}: diverged at token {t}: {hf_ids} vs {vllm_ids}"
            )
            return
    assert len(vllm_ids) == len(hf_ids) == MAX_TOKENS

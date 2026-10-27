# SPDX-License-Identifier: Apache-2.0
"""Greedy generation with text and image inputs must match HF transformers."""

import gc

import pytest
from helpers import (
    DEVICE,
    DTYPE,
    MAX_TOKENS,
    NUM_LOGPROBS,
    assert_matches_hf,
    hf_generate,
    llm_kwargs,
    random_image,
    vllm_prompt,
)
from transformers import AutoProcessor, T5Gemma2ForConditionalGeneration
from vllm import LLM, SamplingParams

IMAGES = [random_image(0, 40, 30), random_image(1, 25, 50)]
CASES = {
    # name: (encoder text, images, decoder prompt)
    "text": ("hello world, this is a text-only prompt!", [], None),
    "one image": ("<start_of_image> what is this?", IMAGES[:1], None),
    "two images": ("two: <start_of_image> and <start_of_image> end", IMAGES, None),
    "decoder prompt": ("explicit decoder prompt", [], "abc"),
}


@pytest.fixture(scope="module")
def llm(model_dir: str):
    llm = LLM(
        model=model_dir, max_num_seqs=4, **llm_kwargs(limit_mm_per_prompt={"image": 2})
    )
    yield llm
    del llm
    gc.collect()


def test_matches_hf(llm: LLM, model_dir: str) -> None:
    """All cases run in one batch, so they also share forward passes."""
    params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        min_tokens=MAX_TOKENS,
        ignore_eos=True,
        logprobs=NUM_LOGPROBS,
    )
    outputs = llm.generate([vllm_prompt(*case) for case in CASES.values()], params)

    processor = AutoProcessor.from_pretrained(model_dir)
    hf_model = T5Gemma2ForConditionalGeneration.from_pretrained(model_dir)
    hf_model = hf_model.to(DEVICE, DTYPE).eval()
    for (name, case), output in zip(CASES.items(), outputs):
        hf_output = hf_generate(hf_model, processor, *case)
        assert_matches_hf(hf_output, output.outputs[0], name)

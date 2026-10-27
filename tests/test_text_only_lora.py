# SPDX-License-Identifier: Apache-2.0
"""Text-only mode, and LoRA on the text encoder and decoder against HF + PEFT."""

import gc

import pytest
import torch
from conftest import REAL_MODEL
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
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, T5Gemma2ForConditionalGeneration
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

pytestmark = pytest.mark.skipif(
    DEVICE == "cuda" and DTYPE == torch.float32,
    reason="vLLM's GPU LoRA kernels need float16 or bfloat16",
)

ATTN = r"self_attn\.(q|k|v|o)_proj"
MLP = r"mlp\.(gate|up|down)_proj"
ADAPTERS = {
    # name: PEFT target modules
    "all": rf"model\.(encoder\.text_model|decoder)\.layers\.\d+\.({ATTN}|{MLP})",
    # Decoder k/v also project the encoder outputs into cross-attention K/V.
    "decoder_kv": r"model\.decoder\.layers\.\d+\.self_attn\.(k|v)_proj",
}
# Random LoRA weights large enough to change the greedy output. On the real
# checkpoint, 0.03 already turns it into noise, where near-ties make vLLM and
# HF diverge at the first token.
LORA_STD = 0.01 if REAL_MODEL else 0.2
TEXT = "hello world, this is a text-only prompt!"
CASES = [
    # (encoder text, decoder prompt, adapter). Model Runner V2 runs requests
    # with a one-token decoder prompt first, so the multi-token one comes first
    # here: the batch order then differs from the scheduling order.
    ("explicit decoder prompt", "abc", "all"),
    (TEXT, None, "all"),
    (TEXT, None, None),
    ("a much longer encoder prompt " * 4, None, "decoder_kv"),
]


def load_hf(model_dir: str):
    model = T5Gemma2ForConditionalGeneration.from_pretrained(model_dir)
    return model.to(DEVICE, DTYPE).eval()


@pytest.fixture(scope="module")
def adapters(model_dir: str, tmp_path_factory: pytest.TempPathFactory):
    paths = {}
    for seed, (name, target) in enumerate(ADAPTERS.items()):
        config = LoraConfig(r=8, lora_alpha=16, target_modules=target)
        model = get_peft_model(load_hf(model_dir), config)
        torch.manual_seed(seed)
        for param_name, param in model.named_parameters():
            if "lora_" in param_name:
                param.data.normal_(0.0, LORA_STD)
        paths[name] = str(tmp_path_factory.mktemp(name))
        model.save_pretrained(paths[name])
    return paths


@pytest.fixture(scope="module")
def llm(model_dir: str):
    llm = LLM(
        model=model_dir,
        max_num_seqs=4,
        **llm_kwargs(
            limit_mm_per_prompt={"image": 0},
            enable_lora=True,
            max_loras=len(ADAPTERS),
            max_lora_rank=8,
        ),
    )
    yield llm
    del llm
    gc.collect()


def test_lora_matches_peft(llm: LLM, model_dir: str, adapters: dict) -> None:
    """Requests with different adapters, or none, share one batch."""
    lora_requests = {
        name: LoRARequest(name, i + 1, path)
        for i, (name, path) in enumerate(adapters.items())
    }
    params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        min_tokens=MAX_TOKENS,
        ignore_eos=True,
        logprobs=NUM_LOGPROBS,
    )
    outputs = llm.generate(
        [vllm_prompt(text, [], decoder_prompt) for text, decoder_prompt, _ in CASES],
        params,
        lora_request=[lora_requests.get(adapter) for *_, adapter in CASES],
    )
    assert outputs[1].outputs[0].token_ids != outputs[2].outputs[0].token_ids, (
        "the adapter does not change the output, so the test proves nothing"
    )

    processor = AutoProcessor.from_pretrained(model_dir)
    for (text, decoder_prompt, adapter), output in zip(CASES, outputs):
        model = load_hf(model_dir)
        if adapter is not None:
            model = PeftModel.from_pretrained(model, adapters[adapter]).eval()
        hf_output = hf_generate(model, processor, text, [], decoder_prompt)
        assert_matches_hf(hf_output, output.outputs[0], f"{adapter=} {text[:20]!r}")


def test_text_only_mode_rejects_images(llm: LLM) -> None:
    prompt = vllm_prompt(
        "<start_of_image> what is this?", [random_image(0, 40, 30)], None
    )
    with pytest.raises(Exception, match="At most 0 image"):
        llm.generate(prompt, SamplingParams(max_tokens=1))

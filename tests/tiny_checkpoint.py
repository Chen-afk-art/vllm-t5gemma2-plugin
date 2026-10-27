# SPDX-License-Identifier: Apache-2.0
"""Build a tiny random T5Gemma 2 checkpoint (model, tokenizer and processor).

It has the real architecture at toy sizes, so tests can compare vLLM with
HF transformers without access to the gated `google/t5gemma-2-*` weights.
"""

import string

import torch
from tokenizers import Tokenizer, decoders, models, processors
from transformers import (
    Gemma3ImageProcessor,
    Gemma3Processor,
    PreTrainedTokenizerFast,
    SiglipVisionConfig,
    T5Gemma2Config,
    T5Gemma2ForConditionalGeneration,
)

SPECIALS = [
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "<start_of_image>",
    "<end_of_image>",
    "<image_soft_token>",
]


def _build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for ch in string.printable:
        vocab.setdefault(ch, len(vocab))
    # vLLM's Gemma 3 processor relies on Gemma's tokens for runs of newlines.
    for n in (2, 3, 4):
        vocab["\n" * n] = len(vocab)
    merges = [("\n", "\n"), ("\n\n", "\n"), ("\n\n", "\n\n")]

    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    tok.add_special_tokens(SPECIALS)
    tok.decoder = decoders.Fuse()
    tok.post_processor = processors.TemplateProcessing(
        single="<bos> $A", pair="<bos> $A $B", special_tokens=[("<bos>", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        extra_special_tokens={
            "boi_token": "<start_of_image>",
            "eoi_token": "<end_of_image>",
            "image_token": "<image_soft_token>",
        },
    )


def _build_config(sliding_window: int) -> T5Gemma2Config:
    text = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=2048,
        sliding_window=sliding_window,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        query_pre_attn_scalar=32,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
        },
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    vision = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=32,
        patch_size=8,
        vision_use_head=False,
    )
    return T5Gemma2Config(
        encoder=dict(
            text_config=text,
            vision_config=vision.to_dict(),
            mm_tokens_per_image=4,
            boi_token_index=4,
            eoi_token_index=5,
        ),
        decoder=text,
        image_token_index=6,
        dropout_rate=0.0,
        attention_dropout=0.0,
        architectures=["T5Gemma2ForConditionalGeneration"],
    )


def build_tiny_checkpoint(out_dir: str, sliding_window: int = 7, seed: int = 0) -> None:
    torch.manual_seed(seed)
    model = T5Gemma2ForConditionalGeneration(_build_config(sliding_window)).eval()
    # HF initializes RMSNorm weights, the EOI vector and the projector to 0;
    # randomize them so that the tests exercise those code paths.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if any(k in name for k in ("norm", "eoi_embedding", "mm_input_projection")):
                param.normal_(0, 0.2)
    generation_config = model.generation_config
    generation_config.decoder_start_token_id = generation_config.bos_token_id = 2
    generation_config.eos_token_id = 1
    generation_config.pad_token_id = 0
    model.save_pretrained(out_dir)

    image_processor = Gemma3ImageProcessor(
        size={"height": 32, "width": 32}, do_pan_and_scan=False
    )
    Gemma3Processor(
        image_processor=image_processor,
        tokenizer=_build_tokenizer(),
        image_seq_length=4,
    ).save_pretrained(out_dir)

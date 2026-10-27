# SPDX-License-Identifier: Apache-2.0
"""Offline inference with T5Gemma 2: text, image, and explicit decoder prompts.

Usage: python examples/offline_inference.py [--model google/t5gemma-2-270m-270m]
"""

import argparse

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset


def main(model: str) -> None:
    llm = LLM(model=model, max_model_len=2048, limit_mm_per_prompt={"image": 1})
    prompts = [
        # Text only: the prompt is the encoder input, the decoder starts at <bos>.
        {"prompt": "Translate English to French: Where is the train station?"},
        # One image; <start_of_image> marks where it goes.
        {
            "prompt": "<start_of_image> in this image, there is",
            "multi_modal_data": {
                "image": ImageAsset("stop_sign").pil_image.convert("RGB")
            },
        },
        # Explicit encoder and decoder prompts.
        {
            "encoder_prompt": {
                "prompt": "<start_of_image> describe the picture.",
                "multi_modal_data": {
                    "image": ImageAsset("cherry_blossom").pil_image.convert("RGB")
                },
            },
            "decoder_prompt": "The picture shows",
        },
    ]
    params = SamplingParams(temperature=0, max_tokens=64)
    for output in llm.generate(prompts, params):
        print(repr(output.outputs[0].text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/t5gemma-2-270m-270m")
    main(parser.parse_args().model)

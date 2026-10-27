# SPDX-License-Identifier: Apache-2.0
"""vLLM plugin for T5Gemma 2 (`T5Gemma2ForConditionalGeneration`)."""

__version__ = "0.1.0"

ARCHITECTURE = "T5Gemma2ForConditionalGeneration"


def register() -> None:
    """Register the model and its config hook; vLLM calls this on startup."""
    from vllm.model_executor.models.registry import ModelRegistry

    from vllm_t5gemma2.config import register_config

    register_config()
    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCHITECTURE, "vllm_t5gemma2.model:T5Gemma2ForConditionalGeneration"
        )


__all__ = ["ARCHITECTURE", "__version__", "register"]

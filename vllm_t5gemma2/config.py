# SPDX-License-Identifier: Apache-2.0
"""Config hook that vLLM runs for T5Gemma 2 before building the engine."""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.config import MODELS_CONFIG_MAP, VerifyAndUpdateConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class T5Gemma2Config(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        from vllm.platforms import current_platform

        lora_config = vllm_config.lora_config
        if lora_config is not None and not lora_config.enable_tower_connector_lora:
            # The text encoder runs on the multimodal encoder path, which only
            # applies LoRA to tower modules when this is enabled.
            logger.info("Enabling `enable_tower_connector_lora` for T5Gemma2 LoRA.")
            lora_config.enable_tower_connector_lora = True

        model_config = vllm_config.model_config
        if not current_platform.is_cuda() or model_config.enforce_eager:
            return

        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

        head_dim = model_config.hf_config.decoder.head_dim
        if get_flash_attn_version(head_size=head_dim) is None:
            # Without FlashAttention, the merged attention falls back to a
            # PyTorch implementation that cannot be captured in CUDA graphs.
            logger.warning(
                "FlashAttention is unavailable on this GPU; running T5Gemma2 "
                "in eager mode with its PyTorch attention, which is slower."
            )
            model_config.enforce_eager = True


def register_config() -> None:
    from vllm_t5gemma2 import ARCHITECTURE

    MODELS_CONFIG_MAP.setdefault(ARCHITECTURE, T5Gemma2Config)

# vllm-t5gemma2

A [vLLM](https://github.com/vllm-project/vllm) plugin for
[T5Gemma 2](https://huggingface.co/collections/google/t5gemma-2)
(`T5Gemma2ForConditionalGeneration`): Google's encoder-decoder model built on
Gemma 3, with text and image input.

Once installed, vLLM loads it automatically through the `vllm.general_plugins`
entry point; there is nothing to import.

## Features

- `google/t5gemma-2-270m-270m`, `google/t5gemma-2-1b-1b` and
  `google/t5gemma-2-4b-4b`.
- **Prompts**: text prompts, any number of images per prompt, and explicit
  `encoder_prompt` / `decoder_prompt` inputs.
- **Text-only mode**: `limit_mm_per_prompt={"image": 0}` skips building and
  loading the SigLIP vision tower.
- **LoRA** on the text encoder and the decoder; adapters and requests without
  one can share a batch.
- **Merged attention**: the decoder attends to its own tokens and the encoder
  outputs with one softmax. This runs on FlashAttention on CUDA (FA2 on
  Blackwell). On CPU, and on GPUs without FlashAttention, it uses a PyTorch
  implementation, and on those GPUs vLLM switches to eager mode.

## Requirements

- **vLLM**: a build with encoder-decoder plugin support, which adds
  `EncDecMultiModalProcessor.process_text_only_encoder_prompt` and
  `CrossAttention(underlying_attn_backend=...)`. Until that is in a release,
  use the `encdec-plugin-support` branch of
  [Chen-afk-art/vllm](https://github.com/Chen-afk-art/vllm/tree/encdec-plugin-support).
  On an older vLLM, the model fails at startup with an error that names the
  missing features.
- **transformers**: 5.0 or later, for the T5Gemma 2 config and processor.
- **Model access**: the `google/t5gemma-2-*` checkpoints are gated. Accept the
  license on Hugging Face and set `HF_TOKEN`.

## Install

```bash
pip install git+https://github.com/Chen-afk-art/vllm-t5gemma2-plugin
```

## Usage

```python
from vllm import LLM, SamplingParams

llm = LLM(model="google/t5gemma-2-270m-270m", limit_mm_per_prompt={"image": 0})
outputs = llm.generate(
    ["Translate English to French: Where is the train station?"],
    SamplingParams(temperature=0, max_tokens=64),
)
print(outputs[0].outputs[0].text)
```

The prompt and its images are the encoder input. The decoder starts from
`<bos>` unless you pass a `decoder_prompt`. Mark the place of each image with
`<start_of_image>`. See
[`examples/offline_inference.py`](examples/offline_inference.py) for image and
explicit decoder prompts.

The OpenAI-compatible server works the same way:

```bash
vllm serve google/t5gemma-2-270m-270m --limit-mm-per-prompt '{"image": 0}'
```

### Text-only mode

`--limit-mm-per-prompt '{"image": 0}'` disables images and does not load the
vision tower. `--language-model-only` is not supported, because the text
encoder runs on vLLM's multimodal encoder path.

### LoRA

```python
from vllm.lora.request import LoRARequest

llm = LLM(model="google/t5gemma-2-270m-270m", enable_lora=True, max_lora_rank=16)
llm.generate(prompts, params, lora_request=LoRARequest("my-adapter", 1, "/path/to/adapter"))
```

- **What it covers**: PEFT adapters on the text encoder and decoder
  projections. LoRA weights of the vision tower are loaded but not applied.
- **Tower LoRA**: the plugin turns on `enable_tower_connector_lora`
  automatically, because vLLM applies LoRA to the text encoder as a multimodal
  "tower".

## Limitations

- No quantized KV cache.
- No pipeline parallelism (vLLM supports it for no encoder-decoder model).
- No LoRA on the vision tower.
- The encoder runs eagerly.
- Chunked prefill is off, as for every encoder-decoder model in vLLM.

## Testing

```bash
pip install -e ".[test]"
pytest tests                                              # tiny random checkpoint
T5GEMMA2_TEST_MODEL=google/t5gemma-2-270m-270m pytest tests  # real checkpoint
```

By default the tests build a tiny random checkpoint with the real
architecture, so they need neither a GPU nor access to the gated weights. They
compare greedy generation with HF transformers (and with HF + PEFT for LoRA)
token by token, mixing requests in one batch.

### Validation so far

On x86 CPU in float32, with tiny random checkpoints:

- **Generation**: identical tokens to HF transformers 5.19 for text, one and
  two images, and explicit decoder prompts. The largest logprob difference is
  5e-7.
- **LoRA**: three random adapters plus requests without an adapter in one
  batch give identical tokens to HF + PEFT.
- **Text-only mode**: the vision tower is not built, and image inputs are
  rejected.
- **OpenAI server**: a completion request returns the same tokens as HF.

Not validated yet: the CUDA paths (FlashAttention, CUDA graphs, Triton LoRA
kernels, Model Runner V2), tensor parallelism, and the real
`google/t5gemma-2-*` checkpoints.

## Acknowledgements

The model code started from vLLM's Gemma 3 and SigLIP implementations and HF
transformers' `modeling_t5gemma2.py`. It was written with AI assistance
(Claude Code).

## License

Apache-2.0

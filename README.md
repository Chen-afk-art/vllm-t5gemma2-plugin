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
  `CrossAttention(underlying_attn_backend=...)`. These are proposed in
  [vllm-project/vllm#64073](https://github.com/vllm-project/vllm/pull/64073).
  Until they are in a release, use that PR's branch, `encdec-plugin-support` of
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
- FlashAttention only runs `float16` and `bfloat16`. In `float32` the model
  uses its PyTorch attention in eager mode on every GPU, which is slower.

## Testing

```bash
pip install -e ".[test]"
pytest tests                                              # tiny random checkpoint
T5GEMMA2_TEST_MODEL=google/t5gemma-2-270m-270m pytest tests  # real checkpoint
# On GPUs without bfloat16 (e.g. T4), add T5GEMMA2_TEST_DTYPE=float16.
```

By default the tests build a tiny random checkpoint with the real
architecture, so they need neither a GPU nor access to the gated weights.

- **What they compare**: greedy generation against HF transformers (and
  against HF + PEFT for LoRA), token by token, with requests mixed in one batch.
- **Dtype**: the tiny checkpoint runs in float32 on CPU and in float16 on GPU,
  where vLLM's LoRA kernels need float16 or bfloat16. Real checkpoints run in
  bfloat16. `T5GEMMA2_TEST_DTYPE` overrides this; with `float32` on GPU, the
  LoRA tests are skipped.
- **Engine process**: the tests run the vLLM engine in the test process
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), so that all requests of a batch arrive
  before its first step. That lets the LoRA test catch a tower LoRA mapping
  built in the wrong order on Model Runner V2.

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

On an NVIDIA A40 (FlashAttention, Model Runner V2):

- **Tiny checkpoint, float16**: all tests pass.
- **Tiny checkpoint, float32**: generation passes with the PyTorch attention in
  eager mode, and so does the merged-attention test. The LoRA tests do not run
  in float32 on GPU, because vLLM's LoRA kernels need float16 or bfloat16.
- **`google/t5gemma-2-270m-270m`, bfloat16**: generation with text, images
  and explicit decoder prompts matches HF, and LoRA matches HF + PEFT. The
  LoRA test passed in a separate run, after lowering its random LoRA scale to
  0.01.
- **Without the Model Runner V2 fix** in vLLM (`[Model Runner V2][LoRA] Build
  the tower LoRA mapping of encoder-decoder models in batch order`), the LoRA
  test fails with both checkpoints: a request's encoder runs with another
  request's adapter. The first mismatch is a request with an adapter on the
  tiny checkpoint, and a request without one on the real checkpoint.

Not validated yet: GPUs without FlashAttention (e.g. T4), Blackwell, and tensor
parallelism. On the A40, `float32` runs the same PyTorch attention that such
GPUs use.

## Acknowledgements

The model code started from vLLM's Gemma 3 and SigLIP implementations and HF
transformers' `modeling_t5gemma2.py`. It was written with AI assistance
(Claude Code).

## License

Apache-2.0

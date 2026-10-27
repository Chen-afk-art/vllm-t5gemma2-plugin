# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

from vllm_t5gemma2.model import _paged_attention_with_lse_torch, _write_kv_cache_torch


def _make_paged_kv(
    keys: list[torch.Tensor], values: list[torch.Tensor], block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Store per-sequence K/V in a shuffled paged cache; return cache, table."""
    num_kv_heads, head_dim = keys[0].shape[1:]
    max_blocks = max((k.shape[0] + block_size - 1) // block_size for k in keys)
    num_blocks = len(keys) * max_blocks
    kv_cache = torch.zeros(num_blocks, num_kv_heads, block_size, 2 * head_dim)
    block_ids = torch.randperm(num_blocks).view(len(keys), max_blocks)
    for i, (k, v) in enumerate(zip(keys, values)):
        pos = torch.arange(k.shape[0])
        slots = block_ids[i, pos // block_size] * block_size + pos % block_size
        _write_kv_cache_torch(kv_cache, k, v, slots)
    return kv_cache, block_ids


@pytest.mark.parametrize("sliding_window", [None, 3])
def test_merged_attention_is_one_softmax(sliding_window: int | None) -> None:
    """Merging the self and cross partial attentions by their LSE must equal a
    single softmax over the concatenated keys, as in HF `T5Gemma2MergedAttention`.
    """
    torch.manual_seed(0)
    num_heads, num_kv_heads, head_dim, block_size = 4, 2, 8, 4
    scale = head_dim**-0.5
    # Two requests: one in its first step with 3 decoder tokens, and one
    # decoding its 6th token. Encoder lengths are 5 and 7.
    dec_lens, query_lens, enc_lens = [3, 6], [3, 1], [5, 7]

    self_k = [torch.randn(n, num_kv_heads, head_dim) for n in dec_lens]
    self_v = [torch.randn(n, num_kv_heads, head_dim) for n in dec_lens]
    cross_k = [torch.randn(n, num_kv_heads, head_dim) for n in enc_lens]
    cross_v = [torch.randn(n, num_kv_heads, head_dim) for n in enc_lens]
    query = torch.randn(sum(query_lens), num_heads, head_dim)
    query_start_loc = torch.tensor([0, *torch.tensor(query_lens).cumsum(0)])

    def partial(keys, values, seq_lens, causal, window):
        kv_cache, block_table = _make_paged_kv(keys, values, block_size)
        return _paged_attention_with_lse_torch(
            query,
            kv_cache,
            query_start_loc,
            torch.tensor(seq_lens),
            block_table,
            causal,
            scale,
            window,
            softcap=None,
        )

    self_out, self_lse = partial(self_k, self_v, dec_lens, True, sliding_window)
    cross_out, cross_lse = partial(cross_k, cross_v, enc_lens, False, None)
    merged = torch.empty_like(query)
    merge_attn_states(merged, self_out, self_lse, cross_out, cross_lse)

    group = num_heads // num_kv_heads
    for i, (q_len, dec_len) in enumerate(zip(query_lens, dec_lens)):
        q = query[query_start_loc[i] : query_start_loc[i + 1]]
        k = torch.cat([self_k[i], cross_k[i]]).repeat_interleave(group, dim=1)
        v = torch.cat([self_v[i], cross_v[i]]).repeat_interleave(group, dim=1)
        q_pos = torch.arange(dec_len - q_len, dec_len)[:, None]
        k_pos = torch.arange(dec_len)[None, :]
        self_mask = k_pos <= q_pos
        if sliding_window is not None:
            self_mask &= k_pos > q_pos - sliding_window
        mask = torch.cat([self_mask, torch.ones(q_len, enc_lens[i]).bool()], dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) * scale
        probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        expected = torch.einsum("hqk,khd->qhd", probs, v)
        torch.testing.assert_close(
            merged[query_start_loc[i] : query_start_loc[i + 1]], expected
        )

"""Memory-lean KV-cache build for TabPFN-3.5 (tabpfn 9.2.0), after hemingweb's S6E10 patch for 9.0.0.

The stock fit_with_cache path runs each ICL layer over every context row at once (full q/k/v, attention
output and MLP activations for all rows), which is what limits the context size. This builds the
keys/values in row chunks, then the attention output and the MLP in row chunks, updating the residual
stream in place. Queries are independent given the keys/values, and the softmax scaling depends only on
the key length, so the result is the same up to floating-point rounding (checked by lean_patch_check.py).

Usage: import lean_patch; lean_patch.apply()   before fitting.
"""
import os

import torch
from tabpfn.architectures import tabpfn_v3_5 as A

CHUNK = int(os.environ.get("LEAN_CHUNK", 16384))
_original_forward = A.ICLTransformerBlock.forward


def _lean_forward(self, x_BRE, single_eval_pos, save_peak_memory_factor=None, *, cached_kv=None, return_kv=False):
    if not return_kv or cached_kv is not None:  # prediction from the cache: unchanged
        return _original_forward(self, x_BRE, single_eval_pos, save_peak_memory_factor,
                                 cached_kv=cached_kv, return_kv=return_kv)
    att = self.icl_attention
    x = x_BRE.contiguous()
    B, R, _ = x.shape
    N = R if single_eval_pos is None else single_eval_pos
    test_heads = att.num_kv_heads_test
    with torch.no_grad():
        K = V = None
        for s in range(0, N, CHUNK):  # keys/values of the context rows, chunk by chunk
            e = min(s + CHUNK, N)
            h = self.layernorm(x[:, s:e])
            k = att.k_norm(att.k_projection(h).view(B, e - s, att.num_kv_heads, att.head_dim))
            v = att.v_projection(h).view(B, e - s, att.num_kv_heads, att.head_dim)
            if K is None:
                K = torch.empty((B, N, att.num_kv_heads, att.head_dim), dtype=v.dtype, device=x.device)
                V = torch.empty_like(K)
            K[:, s:e], V[:, s:e] = k.to(v.dtype), v
            del h, k, v
        with A.recorded_attention_backends() as train_calls:
            for s in range(0, N, CHUNK):  # context rows attend to all context keys
                e = min(s + CHUNK, N)
                xc = x[:, s:e]
                q = att.q_norm(att.q_projection(self.layernorm(xc)).view(B, e - s, att.num_heads, att.head_dim))
                out = A._batched_scaled_dot_product_attention(q, K, V, softmax_scaling_layer=att.softmax_scaling_layer)
                xc.add_(att.out_projection(out.reshape(B, e - s, att.head_dim * att.num_heads)))
                del q, out
        for s in range(N, R, CHUNK):  # any non-context rows use the test KV heads, as in the original
            e = min(s + CHUNK, R)
            xc = x[:, s:e]
            q = att.q_norm(att.q_projection(self.layernorm(xc)).view(B, e - s, att.num_heads, att.head_dim))
            k_, v_ = (K, V) if test_heads is None else (K[:, :, :test_heads], V[:, :, :test_heads])
            out = A._batched_scaled_dot_product_attention(q, k_, v_, softmax_scaling_layer=att.softmax_scaling_layer)
            xc.add_(att.out_projection(out.reshape(B, e - s, att.head_dim * att.num_heads)))
            del q, out
        if test_heads is not None:
            k_cache, v_cache = K[:, :, :test_heads].contiguous(), V[:, :, :test_heads].contiguous()
        else:
            k_cache, v_cache = K, V
        del K, V
        entry = A.KVCacheEntry(key=k_cache.detach(), value=v_cache.detach(),
                               grid_dtype=A.kv_grid_dtype(train_calls) if k_cache.shape[2] == 1 else None)
        for s in range(0, R, CHUNK):  # MLP, chunk by chunk, in place
            xc = x[:, s:min(s + CHUNK, R)]
            xc.add_(self.mlp(self.layernorm_mlp(xc)))
    return x, entry


def apply():
    A.ICLTransformerBlock.forward = _lean_forward

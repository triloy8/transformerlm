from einops import einsum, rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from profiling import nvtx

from transformerlm.models.instrumented.layers import Linear
from leash.inference.sampling import softmax

ALLOWED_ATTENTION_BACKENDS = {"custom", "torch_sdpa"}

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.device = device

        theta_i = theta ** (torch.arange(0, d_k, 2).float() / d_k)
        position = torch.arange(max_seq_len)

        phases = position.unsqueeze(1) / theta_i.unsqueeze(0)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        phases_combined = torch.stack([phases_cos, phases_sin], dim=-1).to(device=device)

        self.register_buffer("phases", phases_combined, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if nvtx.enabled("fine"):
            with nvtx.range("rope/rotate"):
                x = rearrange(x, '... (d_k p) -> ... d_k p', p=2)
                x1 = x[..., 0]
                x2 = x[..., 1]

                phases_cos = self.phases[..., 0][token_positions].to(dtype=x.dtype)
                phases_sin = self.phases[..., 1][token_positions].to(dtype=x.dtype)

                x_rotated = torch.stack([
                    x1 * phases_cos - x2 * phases_sin,
                    x1 * phases_sin + x2 * phases_cos
                ], dim=-1)

                return x_rotated.flatten(-2)
        else:
            x = rearrange(x, '... (d_k p) -> ... d_k p', p=2)
            x1 = x[..., 0]
            x2 = x[..., 1]

            phases_cos = self.phases[..., 0][token_positions].to(dtype=x.dtype)
            phases_sin = self.phases[..., 1][token_positions].to(dtype=x.dtype)

            x_rotated = torch.stack([
                x1 * phases_cos - x2 * phases_sin,
                x1 * phases_sin + x2 * phases_cos
            ], dim=-1)

            return x_rotated.flatten(-2)


def _prepare_attention_mask(attention_mask: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(device=ref_tensor.device, dtype=torch.bool)
    if mask.dim() == 2:
        mask = mask[:, None, None, :]
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    elif mask.dim() != 4:
        raise ValueError("attention_mask must be 2D, 3D, or 4D")
    return mask


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
):
    if nvtx.enabled("fine"):
        with nvtx.range("sdpa/qk"):
            qk_score = einsum(Q, K, "batch_size ... n d_k, batch_size ... m d_k -> batch_size ... n m") / torch.sqrt(torch.tensor(Q.shape[-1]))
            if attention_mask is not None:
                mask = _prepare_attention_mask(attention_mask, qk_score)
                qk_score = qk_score.masked_fill(~mask, float("-inf"))
        with nvtx.range("sdpa/softmax"):
            softmax_qk_score = softmax(qk_score, dim=-1)
        with nvtx.range("sdpa/attnV"):
            attn = einsum(softmax_qk_score, V, "batch_size ... n m, batch_size ... m d_k -> batch_size ... n d_k")
        return attn
    else:
        qk_score = einsum(Q, K, "batch_size ... n d_k, batch_size ... m d_k -> batch_size ... n m") / torch.sqrt(torch.tensor(Q.shape[-1]))
        if attention_mask is not None:
            mask = _prepare_attention_mask(attention_mask, qk_score)
            qk_score = qk_score.masked_fill(~mask, float("-inf"))
        softmax_qk_score = softmax(qk_score, dim=-1)
        attn = einsum(softmax_qk_score, V, "batch_size ... n m, batch_size ... m d_k -> batch_size ... n d_k")
        return attn


def torch_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
):
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    mask = None
    if attention_mask is not None:
        mask = _prepare_attention_mask(attention_mask, Q)
    if nvtx.enabled("fine"):
        with nvtx.range("sdpa/torch"):
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=0.0, is_causal=False)


class MultiheadSelfAttentionRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        attention_backend: str = "custom",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        if attention_backend not in ALLOWED_ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend must be one of {sorted(ALLOWED_ATTENTION_BACKENDS)}")
        self.attention_backend = attention_backend

        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(self.theta, self.d_k, self.max_seq_len, device)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with nvtx.range("attn/q_proj"):
            wqx = self.q_proj(x)
        if nvtx.enabled("fine"):
            with nvtx.range("attn/q_reshape"):
                wqx_rearr = rearrange(wqx, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
            with nvtx.range("attn/q_rope"):
                wqx_rearr_rope = self.rope(wqx_rearr, token_positions)
        else:
            wqx_rearr = rearrange(wqx, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
            wqx_rearr_rope = self.rope(wqx_rearr, token_positions)

        with nvtx.range("attn/k_proj"):
            wkx = self.k_proj(x)
        if nvtx.enabled("fine"):
            with nvtx.range("attn/k_reshape"):
                wkx_rearr = rearrange(wkx, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
            with nvtx.range("attn/k_rope"):
                wkx_rearr_rope = self.rope(wkx_rearr, token_positions)
        else:
            wkx_rearr = rearrange(wkx, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
            wkx_rearr_rope = self.rope(wkx_rearr, token_positions)

        with nvtx.range("attn/v_proj"):
            wvx = self.v_proj(x)
        if nvtx.enabled("fine"):
            with nvtx.range("attn/v_reshape"):
                wvx_rearr = rearrange(wvx, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads, d_v=self.d_v)
        else:
            wvx_rearr = rearrange(wvx, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads, d_v=self.d_v)

        seq_len = token_positions.shape[-1]
        with nvtx.range("attn/sdpa"):
            if self.attention_backend == "torch_sdpa":
                attn = torch_scaled_dot_product_attention(
                    wqx_rearr_rope,
                    wkx_rearr_rope,
                    wvx_rearr,
                    attention_mask=attention_mask,
                )
            else:
                attn = scaled_dot_product_attention(
                    wqx_rearr_rope,
                    wkx_rearr_rope,
                    wvx_rearr,
                    attention_mask=attention_mask,
                )
        if nvtx.enabled("fine"):
            with nvtx.range("attn/merge_heads"):
                attn_rearr = rearrange(attn, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)", num_heads=self.num_heads, d_v=self.d_v)
        else:
            attn_rearr = rearrange(attn, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)", num_heads=self.num_heads, d_v=self.d_v)
        with nvtx.range("attn/out_proj"):
            attn_rearr_proj = self.output_proj(attn_rearr)
        return attn_rearr_proj


class MultiheadCrossAttentionRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        attention_backend: str = "custom",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        if attention_backend not in ALLOWED_ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend must be one of {sorted(ALLOWED_ATTENTION_BACKENDS)}")
        self.attention_backend = attention_backend

        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(self.theta, self.d_k, self.max_seq_len, device)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        token_positions: torch.Tensor,
        context_token_positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with nvtx.range("attn/q_proj"):
            wqx = self.q_proj(x)
        if nvtx.enabled("fine"):
            with nvtx.range("attn/q_reshape"):
                wqx_rearr = rearrange(wqx, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
            with nvtx.range("attn/q_rope"):
                wqx_rearr_rope = self.rope(wqx_rearr, token_positions)
        else:
            wqx_rearr = rearrange(wqx, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
            wqx_rearr_rope = self.rope(wqx_rearr, token_positions)

        with nvtx.range("attn/k_proj"):
            wkx = self.k_proj(context)
        if nvtx.enabled("fine"):
            with nvtx.range("attn/k_reshape"):
                wkx_rearr = rearrange(wkx, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
            with nvtx.range("attn/k_rope"):
                wkx_rearr_rope = self.rope(wkx_rearr, context_token_positions)
        else:
            wkx_rearr = rearrange(wkx, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
            wkx_rearr_rope = self.rope(wkx_rearr, context_token_positions)

        with nvtx.range("attn/v_proj"):
            wvx = self.v_proj(context)
        if nvtx.enabled("fine"):
            with nvtx.range("attn/v_reshape"):
                wvx_rearr = rearrange(wvx, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads, d_v=self.d_v)
        else:
            wvx_rearr = rearrange(wvx, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads, d_v=self.d_v)

        with nvtx.range("attn/sdpa"):
            if self.attention_backend == "torch_sdpa":
                attn = torch_scaled_dot_product_attention(
                    wqx_rearr_rope,
                    wkx_rearr_rope,
                    wvx_rearr,
                    attention_mask=attention_mask,
                )
            else:
                attn = scaled_dot_product_attention(
                    wqx_rearr_rope,
                    wkx_rearr_rope,
                    wvx_rearr,
                    attention_mask=attention_mask,
                )
        if nvtx.enabled("fine"):
            with nvtx.range("attn/merge_heads"):
                attn_rearr = rearrange(attn, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)", num_heads=self.num_heads, d_v=self.d_v)
        else:
            attn_rearr = rearrange(attn, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)", num_heads=self.num_heads, d_v=self.d_v)
        with nvtx.range("attn/out_proj"):
            attn_rearr_proj = self.output_proj(attn_rearr)
        return attn_rearr_proj

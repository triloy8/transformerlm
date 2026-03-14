from einops import einsum, rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformerlm.models.layers import Linear
from trainkit.inference.sampling import softmax

ALLOWED_ATTENTION_BACKENDS = {"custom", "torch_sdpa"}
ALLOWED_SDP_BACKENDS = {"auto", "flash", "mem_efficient", "math"}


def set_sdp_backend(backend: str) -> None:
    backend = backend.lower()
    if backend not in ALLOWED_SDP_BACKENDS:
        raise ValueError(f"attention_sdp_backend must be one of {sorted(ALLOWED_SDP_BACKENDS)}")
    if not torch.cuda.is_available():
        return
    if backend == "auto":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        return
    torch.backends.cuda.enable_flash_sdp(backend == "flash")
    torch.backends.cuda.enable_mem_efficient_sdp(backend == "mem_efficient")
    torch.backends.cuda.enable_math_sdp(backend == "math")


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
    scale = torch.tensor(Q.shape[-1], device=Q.device, dtype=Q.dtype).sqrt()
    qk_score = einsum(Q, K, "batch_size ... n d_k, batch_size ... m d_k -> batch_size ... n m") / scale
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
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=0.0, is_causal=False)


def jvp_safe_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
):
    # Use the explicit attention formulation so torch.func.jvp can differentiate through it.
    return scaled_dot_product_attention(Q, K, V, attention_mask=attention_mask)


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
        jvp_attention: bool = False,
    ) -> torch.Tensor:
        wqx = self.q_proj(x)
        wqx_rearr = rearrange(wqx, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
        wqx_rearr_rope = self.rope(wqx_rearr, token_positions)

        wkx = self.k_proj(x)
        wkx_rearr = rearrange(wkx, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
        wkx_rearr_rope = self.rope(wkx_rearr, token_positions)

        wvx = self.v_proj(x)
        wvx_rearr = rearrange(wvx, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads, d_v=self.d_v)

        if jvp_attention:
            attn = jvp_safe_scaled_dot_product_attention(
                wqx_rearr_rope,
                wkx_rearr_rope,
                wvx_rearr,
                attention_mask=attention_mask,
            )
        elif self.attention_backend == "torch_sdpa":
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
        attn_rearr = rearrange(attn, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)", num_heads=self.num_heads, d_v=self.d_v)
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
        jvp_attention: bool = False,
    ) -> torch.Tensor:
        wqx = self.q_proj(x)
        wqx_rearr = rearrange(wqx, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
        wqx_rearr_rope = self.rope(wqx_rearr, token_positions)

        wkx = self.k_proj(context)
        wkx_rearr = rearrange(wkx, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
        wkx_rearr_rope = self.rope(wkx_rearr, context_token_positions)

        wvx = self.v_proj(context)
        wvx_rearr = rearrange(wvx, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads, d_v=self.d_v)

        if jvp_attention:
            attn = jvp_safe_scaled_dot_product_attention(
                wqx_rearr_rope,
                wkx_rearr_rope,
                wvx_rearr,
                attention_mask=attention_mask,
            )
        elif self.attention_backend == "torch_sdpa":
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
        attn_rearr = rearrange(attn, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)", num_heads=self.num_heads, d_v=self.d_v)
        attn_rearr_proj = self.output_proj(attn_rearr)
        return attn_rearr_proj


class MultiheadSelfAttentionRoPE2D(nn.Module):
    """Self-attention with 2D RoPE using separate row/column rotations."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_height: int,
        max_width: int,
        theta: float,
        attention_backend: str = "custom",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        if self.d_k % 4 != 0:
            raise ValueError("per-head dimension must be divisible by 4 for 2D RoPE")
        self.d_v = self.d_k
        self.theta = theta
        if attention_backend not in ALLOWED_ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend must be one of {sorted(ALLOWED_ATTENTION_BACKENDS)}")
        self.attention_backend = attention_backend

        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

        self.d_k_half = self.d_k // 2
        self.row_rope = RotaryPositionalEmbedding(self.theta, self.d_k_half, int(max_height), device)
        self.col_rope = RotaryPositionalEmbedding(self.theta, self.d_k_half, int(max_width), device)

    def _apply_2d_rope(
        self,
        x: torch.Tensor,
        row_positions: torch.Tensor,
        col_positions: torch.Tensor,
    ) -> torch.Tensor:
        row_part = x[..., : self.d_k_half]
        col_part = x[..., self.d_k_half :]
        row_rot = self.row_rope(row_part, row_positions)
        col_rot = self.col_rope(col_part, col_positions)
        return torch.cat((row_rot, col_rot), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        row_positions: torch.Tensor,
        col_positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        jvp_attention: bool = False,
    ) -> torch.Tensor:
        wqx = self.q_proj(x)
        wqx_rearr = rearrange(
            wqx,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )
        wqx_rearr_rope = self._apply_2d_rope(wqx_rearr, row_positions, col_positions)

        wkx = self.k_proj(x)
        wkx_rearr = rearrange(
            wkx,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )
        wkx_rearr_rope = self._apply_2d_rope(wkx_rearr, row_positions, col_positions)

        wvx = self.v_proj(x)
        wvx_rearr = rearrange(
            wvx,
            "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v",
            num_heads=self.num_heads,
            d_v=self.d_v,
        )

        if jvp_attention:
            attn = jvp_safe_scaled_dot_product_attention(
                wqx_rearr_rope,
                wkx_rearr_rope,
                wvx_rearr,
                attention_mask=attention_mask,
            )
        elif self.attention_backend == "torch_sdpa":
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
        attn_rearr = rearrange(
            attn,
            "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)",
            num_heads=self.num_heads,
            d_v=self.d_v,
        )
        attn_rearr_proj = self.output_proj(attn_rearr)
        return attn_rearr_proj


class MultiheadCrossAttentionRoPE2D(nn.Module):
    """Cross-attention with 2D RoPE using separate row/column rotations."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_height: int,
        max_width: int,
        theta: float,
        attention_backend: str = "custom",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        if self.d_k % 4 != 0:
            raise ValueError("per-head dimension must be divisible by 4 for 2D RoPE")
        self.d_v = self.d_k
        self.theta = theta
        if attention_backend not in ALLOWED_ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend must be one of {sorted(ALLOWED_ATTENTION_BACKENDS)}")
        self.attention_backend = attention_backend

        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

        self.d_k_half = self.d_k // 2
        self.row_rope = RotaryPositionalEmbedding(self.theta, self.d_k_half, int(max_height), device)
        self.col_rope = RotaryPositionalEmbedding(self.theta, self.d_k_half, int(max_width), device)

    def _apply_2d_rope(
        self,
        x: torch.Tensor,
        row_positions: torch.Tensor,
        col_positions: torch.Tensor,
    ) -> torch.Tensor:
        row_part = x[..., : self.d_k_half]
        col_part = x[..., self.d_k_half :]
        row_rot = self.row_rope(row_part, row_positions)
        col_rot = self.col_rope(col_part, col_positions)
        return torch.cat((row_rot, col_rot), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        row_positions: torch.Tensor,
        col_positions: torch.Tensor,
        context_row_positions: torch.Tensor,
        context_col_positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        jvp_attention: bool = False,
    ) -> torch.Tensor:
        wqx = self.q_proj(x)
        wqx_rearr = rearrange(
            wqx,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )
        wqx_rearr_rope = self._apply_2d_rope(wqx_rearr, row_positions, col_positions)

        wkx = self.k_proj(context)
        wkx_rearr = rearrange(
            wkx,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )
        wkx_rearr_rope = self._apply_2d_rope(wkx_rearr, context_row_positions, context_col_positions)

        wvx = self.v_proj(context)
        wvx_rearr = rearrange(
            wvx,
            "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v",
            num_heads=self.num_heads,
            d_v=self.d_v,
        )

        if jvp_attention:
            attn = jvp_safe_scaled_dot_product_attention(
                wqx_rearr_rope,
                wkx_rearr_rope,
                wvx_rearr,
                attention_mask=attention_mask,
            )
        elif self.attention_backend == "torch_sdpa":
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
        attn_rearr = rearrange(
            attn,
            "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)",
            num_heads=self.num_heads,
            d_v=self.d_v,
        )
        return self.output_proj(attn_rearr)

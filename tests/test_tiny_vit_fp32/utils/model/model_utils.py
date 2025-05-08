import math

import torch
from einops import rearrange, einsum, pack, unpack
from torch.amp import autocast
from torch.nn import functional as F

TOKEN_SELF_ATTN_VALUE = -5e4


def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += t.ndim if dim < 0 else 0

    columns = [slice(None)] * t.ndim
    columns[dim] = dim_slice

    return t[tuple(columns)]


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)

    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)

    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(
    freqs,
    t,
    start_index=0,
    scale=1.0,
    seq_dim=-2,
    freqs_seq_dim=None,
):
    dtype = t.dtype

    if freqs_seq_dim is None:
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if (t.ndim == 3) or (freqs_seq_dim is not None):
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim=freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place
    t_transformed = (t_middle * freqs.cos() * scale) + (
        rotate_half(t_middle) * freqs.sin() * scale
    )

    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple

    if m.is_integer():
        return False, tensor

    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2

    return True, F.pad(tensor, (*pad_offset, 0, remainder), value=value)


def l2norm(tensor):
    dtype = tensor.dtype
    normed = F.normalize(tensor, dim=-1)

    return normed.type(dtype)


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = padded_x.unfold(1, forward + backward + 1, 1)

    return tensors.movedim(-1, dim).flatten(dim, dim + 1)


@autocast("cuda", enabled=False)
def apply_rotary_pos_emb(q, k, freqs, scale=1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]

    inv_scale = scale**-1

    if scale.ndim == 2:
        scale = scale[-q_len:, :]

    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale)

    return q, k


def round_down_mult(n, mult):
    return n // mult * mult


def round_up_mult(n, mult):
    return math.ceil(n / mult) * mult


def attend(q, k, v, mask=None, return_sim=False, scale=None):
    scale = scale if scale is not None else q.shape[-1] ** -0.5

    q_heads, k_heads = q.shape[1], k.shape[1]
    num_grouped_queries = q_heads // k_heads

    q = rearrange(q, "b (h qh) ... -> b h qh ...", qh=num_grouped_queries)

    sim = einsum(q, k, "b h qh i d, b h j d -> b h qh i j")
    sim *= scale

    mask_value = -torch.finfo(sim.dtype).max

    if mask is not None:
        sim = sim.masked_fill(~mask, mask_value)

    attn = sim.softmax(dim=-1)

    attn_out = einsum(attn, v, "b h qh i j, b h j d -> b h qh i d")

    attn_out = rearrange(attn_out, "b h qh ... -> b (h qh) ...")

    if not return_sim:
        return attn_out

    sim = rearrange(sim, "b h qh ... -> b (h qh) ...")

    return attn_out, sim


def pack_one_with_inverse(t, pattern):
    packed, ps = pack([t], pattern)

    def inverse(out):
        return unpack(out, ps, pattern)[0]

    return packed, inverse


def interpolate_1d(x, length, mode="bilinear"):
    x, inverse_pack = pack_one_with_inverse(x, "* n")

    x = rearrange(x, "b n -> b 1 n 1")
    x = F.interpolate(x, (length, 1), mode=mode)
    x = rearrange(x, "b 1 n 1 -> b n")

    return inverse_pack(x)


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right

    return F.pad(t, (*zeros, *pad), value=value)

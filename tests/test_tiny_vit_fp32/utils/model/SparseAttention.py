from copy import deepcopy

import einx
import torch
import torch.nn.functional as F
from einops import repeat, reduce, rearrange, einsum
from einops.layers.torch import Rearrange
from model.LocalAttention import LocalAttention
from model.RotaryEmbedding import RotaryEmbedding
from model.model_utils import (
    round_down_mult,
    round_up_mult,
    pad_at_dim,
    attend,
    interpolate_1d,
)
from torch import nn, tensor
from torch.nn.attention.flex_attention import flex_attention


class SparseAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        sliding_window_size,
        compress_block_size,
        selection_block_size,
        num_selected_blocks,
        kv_heads=None,
        num_compressed_mem_kv=1,
        norm=True,
        use_diff_topk=False,
        use_triton_kernel=False,
        interpolated_importance_score=False,
        # If query_heads_share_selected_kv set to True, importance score is averaged across query heads to select
        # top-n buckets of kv per kv head - but can be set to False for each query head within a group
        # to look at different sets of kv buckets. Will need more memory and compute
        query_heads_share_selected_kv=True,
        compress_mlp: nn.Module | None = None,
        compress_mlp_expand_factor=1.0,
        strategy_combine_mlp: nn.Module | None = None,
    ):
        super().__init__()

        # attention heads
        # handling gqa if `kv_heads` is set
        kv_heads = kv_heads if kv_heads is not None else heads
        assert (kv_heads <= heads) and (heads % kv_heads == 0)

        # Initialize attention elements
        self.heads = heads
        self.kv_heads = kv_heads
        self.num_grouped_queries = heads // kv_heads

        # Scale
        self.scale = dim_head**-0.5

        dim_inner = dim_head * heads
        dim_kv_inner = dim_head * kv_heads

        self.norm = nn.RMSNorm(dim) if norm else nn.Identity()

        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(dim_head)

        # qkv
        qkv_split = (dim_inner, dim_kv_inner, dim_kv_inner)

        self.to_qkv = nn.Linear(dim, sum(qkv_split), bias=False)
        self.qkv_split = qkv_split

        # sliding window strategy
        self.sliding_window = LocalAttention(
            dim=dim_head,
            window_size=sliding_window_size,
            causal=True,
            exact_windowsize=True,
            autopad=True,
            use_rotary_pos_emb=False,
        )
        self.sliding_window_size = sliding_window_size

        # compress strategy
        self.compress_block_size = compress_block_size
        assert num_compressed_mem_kv > 0

        self.split_compress_window = Rearrange(
            "b h (w n) d -> b h w n d", n=compress_block_size
        )

        self.num_mem_compress_kv = num_compressed_mem_kv
        self.compress_mem_kv = nn.Parameter(
            torch.zeros(2, kv_heads, num_compressed_mem_kv, dim_head)
        )

        self.k_intrablock_positions = nn.Parameter(
            torch.zeros(kv_heads, compress_block_size, dim_head)
        )

        self.v_intrablock_positions = nn.Parameter(
            torch.zeros(kv_heads, compress_block_size, dim_head)
        )

        if compress_mlp is None:
            compress_dim = compress_block_size * dim_head
            compress_mlp_dim_hidden = int(compress_mlp_expand_factor * compress_dim)

            compress_mlp = nn.Sequential(
                Rearrange("b h w n d -> b h w (n d)"),
                nn.Linear(compress_dim, compress_mlp_dim_hidden),
                nn.ReLU(),
                nn.Linear(compress_mlp_dim_hidden, dim_head),
            )

        self.k_compress = deepcopy(compress_mlp)
        self.v_compress = deepcopy(compress_mlp)

        # selection related
        self.use_diff_topk = use_diff_topk

        # in the case fine block size < compressed block size, will weigh space better when selecting
        self.interpolated_importance_score = interpolated_importance_score
        self.query_heads_share_selected_kv = query_heads_share_selected_kv
        self.selection_block_size = selection_block_size

        assert num_selected_blocks >= 0
        if num_selected_blocks == 0:
            print(
                f"`num_selected_blocks` should be set greater than 0, unless if you are ablating it for experimental purposes"
            )

        self.num_selected_blocks = num_selected_blocks
        self.use_triton_kernel = use_triton_kernel

        # they combine the three sparse branches through a learned combine with sigmoid activation
        if strategy_combine_mlp is None:
            strategy_combine_mlp = nn.Linear(dim, 3 * heads)

            # init to sliding windows first, as network tends to pick up on local patterns first before distant ones
            nn.init.zeros_(strategy_combine_mlp.weight)
            strategy_combine_mlp.bias.data.copy_(tensor([-2.0, -2.0, 2.0] * heads))

        self.to_strategy_combine = nn.Sequential(
            strategy_combine_mlp,
            nn.Sigmoid(),
            Rearrange("b n (h s) -> b h n s", h=heads),
        )

        # split and merging heads
        self.split_heads = Rearrange("b n (h d) -> b h n d", d=dim_head)
        self.merge_heads = Rearrange("b h n d -> b n (h d)")

        # combining heads
        self.combine_heads = nn.Linear(dim_inner, dim, bias=False)

    def forward_inference(self, inp, cache, return_cache=True):
        # destruct cache
        (cache_k, cache_v), (cache_ck, cache_cv) = cache

        # variables
        batch, scale, heads, device = inp.shape[0], self.scale, self.heads, inp.device
        seq_len = cache_k.shape[-2] + 1

        sliding_window = self.sliding_window_size
        compress_divisible_seq_len = round_down_mult(seq_len, self.compress_block_size)

        fine_divisible_seq_len = round_up_mult(seq_len, self.selection_block_size)
        num_fine_blocks = fine_divisible_seq_len // self.selection_block_size

        # maybe prenorm
        inp = self.norm(inp)

        # queries, keys, values
        q, k, v = self.to_qkv(inp).split(self.qkv_split, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))

        # handle cache
        k = torch.cat((cache_k, k), dim=-2)
        v = torch.cat((cache_v, v), dim=-2)

        if return_cache:
            cache_kv = (k, v)
        else:
            cache_kv = None

        # 1. compressed attn inference
        cq = q
        ck = cache_ck
        cv = cache_cv

        repeated_ck = repeat(ck, "b h ... -> b (h gh) ...", gh=self.num_grouped_queries)
        repeated_cv = repeat(cv, "b h ... -> b (h gh) ...", gh=self.num_grouped_queries)

        csim = einsum(q, repeated_ck, "b h i d, b h j d -> b h i j") * scale
        cattn = csim.softmax(dim=-1)

        compressed_attn_out = einsum(cattn, repeated_cv, "b h i j, b h j d -> b h i d")

        if seq_len % self.compress_block_size == 0:
            k_compress_input = self.split_compress_window(
                k[..., -self.compress_block_size :, :] + self.k_intrablock_positions
            )

            v_compress_input = self.split_compress_window(
                v[..., -self.compress_block_size :, :] + self.v_intrablock_positions
            )

            next_ck = self.k_compress(k_compress_input)
            next_cv = self.v_compress(v_compress_input)

            ck = torch.cat((ck, next_ck), dim=-2)
            cv = torch.cat((cv, next_cv), dim=-2)

        if return_cache:
            cache_compressed_kv = (ck, cv)
        else:
            cache_compressed_kv = None

        # 2. fine attention inference (to do - compress and fine diff block sizes)
        assert self.compress_block_size == self.selection_block_size

        importance_scores = csim[..., self.num_mem_compress_kv :]
        importance_scores += torch.randn_like(importance_scores) * 100

        num_compress_blocks = importance_scores.shape[-1]
        num_selected = min(self.num_selected_blocks, num_compress_blocks)
        has_selected_kv_for_fine_attn = num_selected > 0

        # block causal diagonal
        rotated_q, rotated_k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        fine_sliding_window = (seq_len % self.selection_block_size) + 1
        fk = rotated_k[..., -fine_sliding_window:, :]
        fv = v[..., -fine_sliding_window:, :]

        # select out the sparse kv segments as defined by compressed attention map as importance score
        if has_selected_kv_for_fine_attn:
            if self.query_heads_share_selected_kv:
                importance_scores = reduce(
                    importance_scores,
                    "b (h grouped_queries) ... -> b h ...",
                    "mean",
                    grouped_queries=self.num_grouped_queries,
                )

            sel_scores, sel_indices = importance_scores.topk(num_selected, dim=-1)

            fine_divisible_seq_len = round_up_mult(seq_len, self.selection_block_size)
            remainder = fine_divisible_seq_len - k.shape[-2]

            sel_fk = pad_at_dim(rotated_k, (0, remainder), dim=-2)
            sel_fv = pad_at_dim(v, (0, remainder), dim=-2)

            sel_fk = rearrange(
                sel_fk, "b h (w j) d -> b h w j d", j=self.selection_block_size
            )

            sel_fv = rearrange(
                sel_fv, "b h (w j) d -> b h w j d", j=self.selection_block_size
            )

            # get_at('b h [w] j d, b h 1 sel -> b h (sel j) d'

            sel_indices = repeat(
                sel_indices,
                "b h 1 sel -> b h sel j d",
                j=self.selection_block_size,
                d=sel_fk.shape[-1],
            )

            sel_fk = sel_fk.gather(2, sel_indices)
            sel_fv = sel_fv.gather(2, sel_indices)

            sel_fk, sel_fv = tuple(
                rearrange(t, "b h sel j d -> b h (sel j) d") for t in (sel_fk, sel_fv)
            )

            fmask = sel_scores > 1e-10

            fmask = repeat(
                fmask, "b h i sel -> b h i (sel j)", j=self.selection_block_size
            )

            fk = torch.cat((sel_fk, fk), dim=-2)
            fv = torch.cat((sel_fv, fv), dim=-2)

            fmask = F.pad(fmask, (0, fk.shape[-2] - fmask.shape[-1]), value=True)
        else:
            fmask = None

        # remove later
        fq = rearrange(
            rotated_q, "b (h gh) ... -> b h gh ...", gh=self.num_grouped_queries
        )

        fsim = einsum(fq, fk, "b h gh i d, b h j d -> b h gh i j") * scale

        fsim = einx.where(
            "b h i j, b h gh i j, -> b h gh i j",
            fmask,
            fsim,
            -torch.finfo(fsim.dtype).max,
        )

        fattn = fsim.softmax(dim=-1)

        fine_attn_out = einsum(fattn, fv, "b h gh i j, b h j d -> b h gh i d")
        fine_attn_out = rearrange(fine_attn_out, "b h gh ... -> b (h gh) ...")

        # 3. sliding window
        k = repeat(k, "b h ... -> b (h gh) ...", gh=self.num_grouped_queries)
        v = repeat(v, "b h ... -> b (h gh) ...", gh=self.num_grouped_queries)

        sliding_slice = (Ellipsis, slice(-(sliding_window + 1), None), slice(None))
        rotated_q, rotated_k = self.rotary_emb.rotate_queries_with_cached_keys(
            q, k[sliding_slice]
        )

        sim = einsum(rotated_q, rotated_k, "b h i d, b h j d -> b h i j") * scale
        attn = sim.softmax(dim=-1)
        sliding_window_attn_out = einsum(
            attn, v[sliding_slice], "b h i j, b h j d -> b h i d"
        )

        # combine strategies
        strategy_weighted_combine = self.to_strategy_combine(inp)

        out = einsum(
            strategy_weighted_combine,
            torch.stack(
                [
                    compressed_attn_out,
                    fine_attn_out,
                    sliding_window_attn_out,
                ]
            ),
            "b h n s, s b h n d -> b h n d",
        )

        # merge heads and combine them
        out = self.merge_heads(out)
        out = self.combine_heads(out)

        if not return_cache:
            return out

        return out, (cache_kv, cache_compressed_kv)

    def forward(
        self,
        inp,
        cache=None,
        disable_triton_kernel=False,
        sliding_window_flex_mask=None,
        fine_selection_flex_mask=None,
        return_cache=False,
    ):
        is_inferencing = cache is not None

        if is_inferencing:
            assert (
                inp.shape[1] == 1
            ), "input must be single tokens if inferencing with cache key values"
            return self.forward_inference(inp, cache, return_cache=return_cache)

        batch, seq_len, scale, heads, device = (
            *inp.shape[:2],
            self.scale,
            self.heads,
            inp.device,
        )

        compress_divisible_seq_len = round_down_mult(seq_len, self.compress_block_size)
        num_compress_blocks = compress_divisible_seq_len // self.compress_block_size

        fine_divisible_seq_len = round_up_mult(seq_len, self.selection_block_size)
        num_fine_blocks = fine_divisible_seq_len // self.selection_block_size

        # maybe prenorm
        inp = self.norm(inp)

        # queries, keys, values
        q, k, v = self.to_qkv(inp).split(self.qkv_split, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))

        # handle cache
        if return_cache:
            cache_kv = (k, v)
        else:
            cache_kv = None

        # compressed key / values - variables prepended with `c` stands for compressed
        k_pos = repeat(
            self.k_intrablock_positions, "h n d -> h (r n) d", r=num_compress_blocks
        )

        v_pos = repeat(
            self.v_intrablock_positions, "h n d -> h (r n) d", r=num_compress_blocks
        )

        k_compress_input = self.split_compress_window(
            k[..., :compress_divisible_seq_len, :] + k_pos
        )

        v_compress_input = self.split_compress_window(
            v[..., :compress_divisible_seq_len, :] + v_pos
        )

        cq = q

        ck = self.k_compress(
            k_compress_input
        )  # Equation (7) of the Native Sparse Attention paper

        cv = self.v_compress(v_compress_input)

        if return_cache:
            cache_compressed_kv = (ck, cv)
        else:
            cache_compressed_kv = None

        # 1. coarse attention over compressed
        mem_ck, mem_cv = repeat(self.compress_mem_kv, "kv ... -> kv b ...", b=batch)
        num_mem_compress_kv = mem_ck.shape[-2]

        ck = torch.cat((mem_ck, ck), dim=-2)
        cv = torch.cat((mem_cv, cv), dim=-2)

        cq_seq = torch.arange(seq_len, device=device)

        ck_seq = (
            (torch.arange(num_compress_blocks, device=device) + 1)
            * self.compress_block_size
        ) - 1

        ck_seq = F.pad(ck_seq, (num_mem_compress_kv, 0), value=-1)

        cmask = einx.less("j, i -> i j", ck_seq, cq_seq)
        compressed_attn_out, csim = attend(cq, ck, cv, mask=cmask, return_sim=True)

        # for 2. and 3., will give them relative positions with rotary - compressed needs to be handled separately
        # (even if they already have intra block absolute positions)
        rotated_q, rotated_k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # 2. fine attention over selected based on compressed attention logits -
        # variables prepended with `f` stands for the fine attention pathway
        importance_scores = csim[..., num_mem_compress_kv:]

        num_selected = min(self.num_selected_blocks, num_compress_blocks)
        has_selected_kv_for_fine_attn = num_selected > 0

        # maybe average the compressed attention across each grouped queries (per key / values)
        if self.query_heads_share_selected_kv:
            importance_scores = reduce(
                importance_scores,
                "b (h grouped_queries) ... -> b h ...",
                "mean",
                grouped_queries=self.num_grouped_queries,
            )

            fine_num_grouped_queries = self.num_grouped_queries
        else:
            fine_num_grouped_queries = 1

        # handle if compress block size does not equal to the fine block size
        # cannot parse their equation, so will just improvise
        # first we expand all the compressed scores to the full sequence length,
        # then average within each fine / selection block size - pad on the right to 0s,
        # which should be fine as the sliding window covers the local anyway
        if has_selected_kv_for_fine_attn:

            if self.compress_block_size != self.selection_block_size:
                compress_seq_len = num_compress_blocks * self.compress_block_size

                if self.interpolated_importance_score:
                    importance_scores = interpolate_1d(
                        importance_scores, compress_seq_len
                    )
                else:
                    importance_scores = repeat(
                        importance_scores,
                        "... j -> ... (j block_size)",
                        block_size=self.compress_block_size,
                    )

                padding = fine_divisible_seq_len - compress_seq_len

                fine_query_seq_len = importance_scores.shape[-2]
                fine_query_padding = (
                    fine_divisible_seq_len - importance_scores.shape[-2]
                )

                importance_scores = F.pad(importance_scores, (0, padding))

                # mask out the diagonal since block causal is included by default for fine attending
                block_causal_mask = torch.ones(
                    (num_fine_blocks,) * 2, device=device, dtype=torch.bool
                ).tril(-1)

                block_causal_mask = repeat(
                    block_causal_mask,
                    "i j -> (i n1) (j n2)",
                    n1=self.selection_block_size,
                    n2=self.selection_block_size,
                )

                block_causal_mask = block_causal_mask[:fine_query_seq_len]

                importance_scores = importance_scores.masked_fill(
                    ~block_causal_mask, -torch.finfo(csim.dtype).max
                )

                importance_scores = reduce(
                    importance_scores,
                    "... (j block_size) -> ... j",
                    "mean",
                    block_size=self.selection_block_size,
                )

            importance_scores = F.pad(importance_scores, (1, 0), value=-1e3)
            importance_scores = importance_scores.softmax(dim=-1)
            importance_scores = importance_scores[..., 1:]

        # handle if number of total blocks is less than number to select for fine attention
        fq = rotated_q
        fk = rotated_k
        fv = v

        if has_selected_kv_for_fine_attn:
            # get the top-n kv segments for fine attention
            selected_importance_values, selected_block_indices = importance_scores.topk(
                min(num_selected, importance_scores.shape[-1]), dim=-1
            )

            gates = None

            if self.use_diff_topk:
                gates = (
                    selected_importance_values
                    + (1.0 - selected_importance_values).detach()
                )

            if fine_selection_flex_mask is not None:
                assert (
                    not self.use_diff_topk
                ), "differential topk is not available for flex attention"

                # flex attention for the selection for fine attention
                fine_block_mask = fine_selection_flex_mask(
                    selected_block_indices, num_grouped_queries=fine_num_grouped_queries
                )

                fine_attn_out = flex_attention(
                    fq, fk, fv, block_mask=fine_block_mask, enable_gqa=True
                )
            else:
                fmask = selected_importance_values > 1e-10

                if seq_len < fine_divisible_seq_len:
                    remainder = fine_divisible_seq_len - seq_len
                    fk = pad_at_dim(fk, (0, remainder), value=0.0, dim=-2)
                    fv = pad_at_dim(fv, (0, remainder), value=0.0, dim=-2)
                    fq = pad_at_dim(fq, (0, remainder), value=0.0, dim=-2)

                    fmask = pad_at_dim(fmask, (0, remainder), value=False, dim=-2)

                    selected_block_indices = pad_at_dim(
                        selected_block_indices, (0, remainder), value=0, dim=-2
                    )

                    if gates is not None:
                        gates = pad_at_dim(gates, (0, remainder), value=0, dim=-2)

                # handle block causal diagonal in the diagram, but run experiments without to see
                fine_window_seq = (
                    torch.arange(fine_divisible_seq_len, device=device)
                    // self.selection_block_size
                )

                fine_window_seq = repeat(
                    fine_window_seq,
                    "n -> b h n 1",
                    b=batch,
                    h=selected_block_indices.shape[1],
                )

                selected_block_indices = torch.cat(
                    (selected_block_indices, fine_window_seq), dim=-1
                )  # for the block causal diagonal in fig2

                fmask = repeat(
                    fmask, "b h i w -> b h i w j", j=self.selection_block_size
                )

                causal_mask = torch.ones(
                    (self.selection_block_size,) * 2, device=device, dtype=torch.bool
                ).tril()

                causal_mask = repeat(
                    causal_mask,
                    "i j -> b h (w i) 1 j",
                    w=num_fine_blocks,
                    b=batch,
                    h=fmask.shape[1],
                )

                fmask = torch.cat((fmask, causal_mask), dim=-2)
                fmask = rearrange(fmask, "b h i w j -> b h i (w j)")

                # select out the spatial crops of keys / values for fine attention
                fk = rearrange(fk, "b h (w n) d -> b h w n d", w=num_fine_blocks)
                fv = rearrange(fv, "b h (w n) d -> b h w n d", w=num_fine_blocks)

                # get_at("b h [w] j d, b h i selected -> b h i selected j d", fkv, selected_block_indices)
                if self.query_heads_share_selected_kv:
                    fk = repeat(
                        fk,
                        "b h w j d -> b h i w j d",
                        i=selected_block_indices.shape[2],
                    )

                    fv = repeat(
                        fv,
                        "b h w j d -> b h i w j d",
                        i=selected_block_indices.shape[2],
                    )
                else:
                    fk = repeat(
                        fk,
                        "b h w j d -> b (h qh) i w j d",
                        i=selected_block_indices.shape[2],
                        qh=self.num_grouped_queries,
                    )

                    fv = repeat(
                        fv,
                        "b h w j d -> b (h qh) i w j d",
                        i=selected_block_indices.shape[2],
                        qh=self.num_grouped_queries,
                    )

                selected_block_indices = repeat(
                    selected_block_indices,
                    "b h i sel -> b h i sel j d",
                    j=fk.shape[-2],
                    d=fk.shape[-1],
                )

                fk = fk.gather(3, selected_block_indices)
                fv = fv.gather(3, selected_block_indices)

                # differential topk gating
                if self.use_diff_topk:
                    gates = F.pad(gates, (0, 1), value=1.0)
                    fk = einx.multiply(
                        "b h i sel, b h i sel j d -> b h i sel j d", gates, fk
                    )

                # merge selected key values
                fk, fv = tuple(
                    rearrange(t, "b h i w j d -> b h i (w j) d") for t in (fk, fv)
                )

                # fine attention
                fmask = rearrange(fmask, "b h ... -> b h 1 ...")

                fq = rearrange(
                    fq, "b (h qh) ... -> b h qh ...", qh=fine_num_grouped_queries
                )

                fsim = (
                    einsum(fq, fk, "b h qh i d, b h i j d -> b h qh i j") * self.scale
                )

                mask_value = -torch.finfo(fsim.dtype).max
                fsim = fsim.masked_fill(~fmask, mask_value)
                fattn = fsim.softmax(dim=-1)
                fine_attn_out = einsum(fattn, fv, "b h qh i j, b h i j d -> b h qh i d")
                fine_attn_out = rearrange(fine_attn_out, "b h qh ... -> b (h qh) ...")
                fine_attn_out = fine_attn_out[..., :seq_len, :]
        else:
            # if only first block, just do a simple block causal
            seq_len = fk.shape[-2]

            fmask = causal_mask = torch.ones(
                (seq_len, seq_len), device=device, dtype=torch.bool
            ).tril()

            fine_attn_out = attend(fq, fk, fv, mask=fmask)

        # 3. overlapping sliding window, this is unsurprising and expected - `s` for sliding
        sq = rotated_q
        sk = rotated_k
        sv = v

        if sliding_window_flex_mask is not None:
            sliding_window_attn_out = flex_attention(
                sq, sk, sv, block_mask=sliding_window_flex_mask, enable_gqa=True
            )
        else:
            sk, sv = tuple(
                repeat(
                    t,
                    "b h ... -> b (h num_grouped_queries) ...",
                    num_grouped_queries=self.num_grouped_queries,
                )
                for t in (sk, sv)
            )

            sliding_window_attn_out = self.sliding_window(sq, sk, sv)

        # combine strategies
        strategy_weighted_combine = self.to_strategy_combine(inp)
        out = einsum(
            strategy_weighted_combine,
            torch.stack([compressed_attn_out, fine_attn_out, sliding_window_attn_out]),
            "b h n s, s b h n d -> b h n d",
        )

        # merge heads and combine them
        out = self.merge_heads(out)
        out = self.combine_heads(out)

        if not return_cache:
            return out

        return out, (cache_kv, cache_compressed_kv)

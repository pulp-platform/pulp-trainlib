import torch.nn as nn

from utils.vit_lr.MultiHeadSelfAttention import MultiHeadSelfAttention
from utils.vit_lr.PositionWiseFeedForward import PositionWiseFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, tgt_len, ff_dim, dropout, device):
        super().__init__()
        self.tgt_len = tgt_len

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            n_heads=num_heads,
            att_dim=dim,
            device=device,
        )

        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)

    def forward(self, x):
        h = self.norm1(x)
        h = self.attn(h, self.tgt_len)
        h = self.proj(h)
        h = self.drop(h)

        x = x + h

        h = self.norm2(x)
        h = self.pwff(h)
        h = self.drop(h)

        x = x + h

        return x

    def get_model_graph_information(self, x, name):
        # Input shape: b, nph * npw + 1, dim
        all_nodes = dict()
        ordered_nodes = []

        # ================== norm_1 ==================
        h = self.norm1(x)
        all_nodes[name + "_norm1"] = {
            "input": "concat_output_data",
            "eps": self.norm1.eps,
            "shape": tuple(h.shape[1:]),
        }
        ordered_nodes.append(name + "_norm1")

        # ================== pre_attn_transpose ==================
        # Necessary pre-attention transpose (due to TrainLib mhsa layer implementation)
        pre_transpose_shape = tuple(h.shape[1:])[::-1]
        all_nodes[name + "_pre_attn_transpose"] = {
            "input_from": name + "_norm1",
            "available_input": False,
            "input_shape": tuple(h.shape[1:]),
            "output_shape": pre_transpose_shape,
        }
        ordered_nodes.append(name + "_pre_attn_transpose")

        # ================== attn ==================
        in_h, in_w = h.shape[-2:]
        att_dim = self.attn.att_dim
        n_heads = self.attn.n_heads

        shape_a = (att_dim, in_h)
        shape_b = (in_h, in_h, n_heads)
        shape_c = (in_w, in_h)

        shape_temp = (1, 1, max(in_h * in_h, in_h * (att_dim // n_heads)))
        shape_sm = (n_heads, in_h, in_h)

        h = self.attn(h, self.tgt_len)
        all_nodes[name + "_attn"] = {
            "q_shape": shape_a,
            "k_shape": shape_a,
            "v_shape": shape_a,
            "att_map_shape": shape_a,
            "softmax_buffer_shape": shape_b,
            "output_shape": shape_c,
            "n_heads": n_heads,
            "input_shape": shape_c,
            "temp_shape": shape_temp,
            "sm_shape": shape_sm,
            "wgt_in_q_shape": tuple(self.attn.proj_q.weight.shape),
            "wgt_in_k_shape": tuple(self.attn.proj_k.weight.shape),
            "wgt_in_v_shape": tuple(self.attn.proj_v.weight.shape),
            "bias_in_q_shape": tuple(self.attn.proj_q.bias.shape),
            "bias_in_k_shape": tuple(self.attn.proj_k.bias.shape),
            "bias_in_v_shape": tuple(self.attn.proj_v.bias.shape),
            "wgt_proj_out_shape": tuple(self.attn.proj_out.weight.shape),
            "input": name + "_pre_attn_transpose_output_data",
        }
        ordered_nodes.append(name + "_attn")

        # ================== post_attn_transpose ==================
        # Necessary post-attention transpose (due to TrainLib mhsa layer implementation)
        post_transpose_shape = tuple(h.shape)[::-1]
        all_nodes[name + "_post_attn_transpose"] = {
            "input_from": name + "_attn",
            "available_input": False,
            "input_shape": post_transpose_shape,
            "output_shape": tuple(h.shape),
        }
        ordered_nodes.append(name + "_post_attn_transpose")

        # ================== proj ==================
        previous_shape = tuple(h.shape)
        h = self.proj(h)
        all_nodes[name + "_proj"] = {
            "input_a": name + "_post_attn_transpose_output_data",
            "input_b": (name + "_proj_weight").upper(),
            "input_a_shape": tuple(previous_shape),
            "bias_shape": tuple(self.proj.bias.shape),
            "output_shape": tuple(h.shape),
        }
        ordered_nodes.append(name + "_proj")

        h = self.drop(h)
        # TODO: Add dropout node

        # ================== add_1 ==================
        x = x + h
        all_nodes[name + "_add_1"] = {
            "input_from": ["concat_output_data", name + "_proj_output_data"],
            "available_input": [False, False],
            "shape": tuple(x.shape),
        }
        ordered_nodes.append(name + "_add_1")

        # ================== norm_2 ==================
        h = self.norm2(x)
        all_nodes[name + "_norm2"] = {
            "input": "concat_output_data",
            "eps": self.norm2.eps,
            "shape": tuple(h.shape[1:]),
        }
        ordered_nodes.append(name + "_norm2")

        # ================== pwff ==================
        h, pwff_all_nodes, pwff_ordered_nodes = self.pwff.get_model_graph_information(h, name + "_pwff")
        all_nodes = {**all_nodes, **pwff_all_nodes}
        ordered_nodes += pwff_ordered_nodes

        h = self.drop(h)
        # TODO: Add dropout node

        # ================== add_2 ==================
        x = x + h
        all_nodes[name + "_add_2"] = {
            "input_from": ["concat_output_data", name + "_pwff_fc2_output_data"],
            "available_input": [False, False],
            "shape": tuple(x.shape),
        }
        ordered_nodes.append(name + "_add_2")

        return x, all_nodes, ordered_nodes

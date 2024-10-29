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

        # ================== attn ==================
        h = self.attn(h, self.tgt_len)
        all_nodes[name + "_attn"] = {
            "q_shape": tuple(h.shape[-2:]),
            "k_shape": tuple(h.shape[-2:]),
            "v_shape": tuple(h.shape[-2:]),
            "att_map_shape": tuple(h.shape[-2:]),
            "softmax_buffer_shape": tuple(h.shape[-2:]),
            "output_shape": tuple(h.shape[-2:]),
            "n_heads": self.attn.n_heads,
            "input_shape": tuple(h.shape[-2:]),
            "wgt_in_q_shape": tuple(self.attn.proj_q.weight.shape),
            "wgt_in_k_shape": tuple(self.attn.proj_k.weight.shape),
            "wgt_in_v_shape": tuple(self.attn.proj_v.weight.shape),
            "bias_in_q_shape": tuple(self.attn.proj_q.bias.shape),
            "bias_in_k_shape": tuple(self.attn.proj_k.bias.shape),
            "bias_in_v_shape": tuple(self.attn.proj_v.bias.shape),
            "wgt_proj_out_shape": tuple(self.attn.proj_out.weight.shape),
            "input": name + "_norm1_output_data",
        }
        ordered_nodes.append(name + "_attn")

        # ================== proj ==================
        h = self.proj(h)
        all_nodes[name + "_proj"] = {
            "input": name + "_attn_out_data",
            "input_shape": tuple(h.shape[-2:]),
            "weight_shape": tuple(self.proj.weight.shape),
            "bias_shape": tuple(self.proj.bias.shape),
            "output_shape": tuple(h.shape[-2:]),
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

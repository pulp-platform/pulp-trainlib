import torch
import torch.nn as nn

from utils.vit_lr.PositionalEmbedding1D import PositionalEmbedding1D
from utils.vit_lr.Transformer import Transformer


# TODO: Update ViTLR model in: the main implementation here, CLion implementation
class ViTLR(nn.Module):
    def __init__(
            self,
            device,
            input_size: tuple[int, int] = (128, 128),
            in_channels: int = 3,
            hidden_dimension: int = 768,
            patch_size: tuple[int, int] = (16, 16),
            num_blocks: int = 12,
            num_heads: int = 12,
            ff_dim: int = 3072,
            dropout_rate: float = 0.1,
            num_classes: int = 1000,
            # Currently supports choosing only one of the transformer blocks or -1 for native run
            latent_replay_layer: int = -1,
    ):
        super().__init__()

        # Check if the input image can be split into an exact number of patches of given size
        assert (
                input_size[0] % patch_size[0] == 0
        ), "Incompatible first shape of input and patch sizes."
        assert (
                input_size[1] % patch_size[1] == 0
        ), "Incompatible second shape of input and patch sizes."

        # Generate required layers
        self.seq_len = int(
            (input_size[0] / patch_size[0]) * (input_size[1] / patch_size[1]) + 1
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dimension))
        self.positional_embedding = PositionalEmbedding1D(
            self.seq_len, hidden_dimension
        )

        self.transformer = Transformer(
            num_blocks=num_blocks,
            dim=hidden_dimension,
            num_heads=num_heads,
            tgt_len=self.seq_len,
            ff_dim=ff_dim,
            dropout=dropout_rate,
            latent_replay_block=latent_replay_layer,
            device=device,
        )

        self.norm = nn.LayerNorm(hidden_dimension, eps=1e-6)
        self.fc = nn.Linear(hidden_dimension, num_classes)

        self.ordered_nodes = None
        self.all_nodes = None

    def forward(self, x, get_activation=False):
        # Check whether the input is a pattern (an original image), or a stored activation
        if isinstance(x, tuple):
            is_pattern, x = x
        else:
            is_pattern = True

        # Store activation if passed
        activation = None
        if get_activation and not is_pattern:
            activation = x.clone().detach()

        if is_pattern:
            b, c, h, w = x.shape

            # b, c, h, w
            x = self.patch_embedding(x)

            # b, dim, nph, npw (number of patches - height and width)
            x = x.flatten(2).transpose(1, 2)

            # b, nph * npw, dim
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)

            # b, nph * npw + 1, dim
            x = self.positional_embedding(x)

        # b, nph * npw + 1, dim
        assert (
                x.shape[1] == self.seq_len
        ), f"Expected activation with second shape {self.seq_len}, got {x.shape[1]}."
        assert x.shape[2] == self.patch_embedding.out_channels, (
            f"Expected activation with third shape {self.patch_embedding.out_channels}"
            f", got {x.shape[2]}."
        )
        if get_activation and is_pattern:
            x, activation = self.transformer(x=(is_pattern, x), get_activation=True)
        else:
            x = self.transformer(x=(is_pattern, x), get_activation=False)

        # b, nph * npw + 1, dim
        x = torch.tanh(x)

        # b, nph * npw + 1, dim
        x = self.norm(x)[:, 0]

        # b, dim
        x = self.fc(x)

        # b, num_classes
        if not get_activation:
            return x
        else:
            return x, activation

    def set_backbone_requires_grad(
            self, trainable: bool, only_before_lr_layer: bool = False
    ):
        self.patch_embedding.requires_grad_(trainable)
        self.class_token.requires_grad_(trainable)
        self.positional_embedding.requires_grad_(trainable)

        if only_before_lr_layer:
            for i, transformer_block in enumerate(self.transformer.blocks):
                if i == self.transformer.latent_replay_block:
                    break
                transformer_block.requires_grad_(trainable)
        else:
            self.transformer.requires_grad_(trainable)
            self.norm.requires_grad_(trainable)

        for transformer_block in self.transformer.blocks:
            transformer_block.attn.proj_out.requires_grad_(False)
            if transformer_block.attn.proj_out.bias is not None:
                transformer_block.attn.proj_out.bias.requires_grad_(False)

        self.set_layer_norm_trainable()

    def set_layer_norm_trainable(self):
        for transformer_block in self.transformer.blocks:
            transformer_block.norm1.requires_grad_(True)
            transformer_block.norm2.requires_grad_(True)

    def get_model_graph_information(self, x, get_activation=False):
        # TODO: Change into hooks
        self.ordered_nodes = list()
        self.all_nodes = dict()

        # Check whether the input is a pattern (an original image), or a stored activation
        if isinstance(x, tuple):
            is_pattern, x = x
        else:
            is_pattern = True

        # Store activation if passed
        activation = None
        if get_activation and not is_pattern:
            activation = x.clone().detach()

        if is_pattern:
            # ================== input ==================
            b, c, h, w = x.shape

            self.all_nodes["input"] = {
                "output_shape": tuple(x.shape[1:]),
            }
            self.ordered_nodes.append("input")
            next_input_shape = tuple(x.shape[1:])

            # ================== patch_embedding ==================
            # b, c, h, w
            x = self.patch_embedding(x)

            self.all_nodes["patch_embedding"] = {
                "input_from": "input",
                "available_input": True,
                "weights_shape": tuple(self.patch_embedding.weight.shape),
                "bias_shape": tuple(self.patch_embedding.bias.shape),
                "input_shape": next_input_shape,
                "output_shape": tuple(x.shape[1:]),

                "stride_h": self.patch_embedding.stride[0],
                "stride_w": self.patch_embedding.stride[1],
            }
            self.ordered_nodes.append("patch_embedding")
            next_input_shape = tuple(x.shape[1:])

            # ================== flatten_and_transpose ==================
            # b, dim, nph, npw (number of patches - height and width)
            x = x.flatten(2).transpose(1, 2)

            # This works because b will always be 1 (working with "virtual" batch sizes). TODO: Caution if this changes!
            self.all_nodes["flatten_and_transpose"] = {
                "input_from": "patch_embedding",
                "available_input": False,
                "input_shape": next_input_shape,
                "output_shape": tuple(x.shape[1:]),
            }
            self.ordered_nodes.append("flatten_and_transpose")
            next_input_shape = tuple(x.shape[1:])

            # ================== concat ==================
            # b, nph * npw, dim
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)

            # This works because b will always be 1 (working with "virtual" batch sizes). TODO: Caution if this changes!
            self.all_nodes["concat"] = {
                "input_from": ["CLASS_TOKEN", "flatten_and_transpose"],
                "available_input": [True, False],
                "input_shape": [tuple(self.class_token.expand(b, -1, -1).shape), next_input_shape],
                "output_shape": tuple(x.shape[1:]),
            }
            self.ordered_nodes.append("concat")

            # ================== positional_embedding ==================
            # b, nph * npw + 1, dim
            x = self.positional_embedding(x)

            self.all_nodes["positional_embedding"] = {
                "input_from": ["concat", "POSITIONAL_EMBEDDING_POS_EMBEDDING"],
                "available_input": [False, True],
                "shape": tuple(x.shape),
            }
            self.ordered_nodes.append("positional_embedding")

        # ================== transformer ==================
        # b, nph * npw + 1, dim
        assert (
                x.shape[1] == self.seq_len
        ), f"Expected activation with second shape {self.seq_len}, got {x.shape[1]}."
        assert x.shape[2] == self.patch_embedding.out_channels, (
            f"Expected activation with third shape {self.patch_embedding.out_channels}"
            f", got {x.shape[2]}."
        )
        if get_activation and is_pattern:
            x, activation, all_nodes_transformer, ordered_nodes_transformer = self.transformer.get_model_graph_information(x=(is_pattern, x), get_activation=True)
        else:
            x, all_nodes_transformer, ordered_nodes_transformer = self.transformer.get_model_graph_information(x=(is_pattern, x), get_activation=False)

        self.all_nodes = self.all_nodes | all_nodes_transformer
        self.ordered_nodes += ordered_nodes_transformer

        # b, nph * npw + 1, dim
        x = torch.tanh(x)
        self.all_nodes["tanh"] = {
            "input": self.all_nodes[self.ordered_nodes[-1]]["input_from"][0],
            "shape": tuple(x.shape),
        }
        self.ordered_nodes.append("tanh")

        # b, nph * npw + 1, dim
        x = self.norm(x)[:, 0]
        self.all_nodes["norm"] = {
            "shape": tuple(x.shape),
            "eps": self.norm.eps,
            "input": self.ordered_nodes[-1] + "_output_data",
        }
        self.ordered_nodes.append("norm")
        previous_shape = tuple(x.shape)

        # b, dim
        x = self.fc(x)
        self.all_nodes["fc"] = {
            "input_a": self.ordered_nodes[-1] + "_output_data",
            "input_b": "FC_WEIGHT",
            "input_a_shape": tuple(previous_shape),
            "bias_shape": tuple(self.fc.bias.shape),
            "output_shape": tuple(x.shape),
        }
        self.ordered_nodes.append("fc")

        # b, num_classes
        # if not get_activation:
        #     return x
        # else:
        #     return x, activation

        return self.ordered_nodes, self.all_nodes

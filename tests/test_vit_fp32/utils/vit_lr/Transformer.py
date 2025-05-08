import torch.nn as nn

from utils.vit_lr.TransformerBlock import TransformerBlock


class Transformer(nn.Module):
    def __init__(
        self,
        num_blocks,
        dim,
        num_heads,
        tgt_len,
        ff_dim,
        dropout,
        latent_replay_block,
        device,
    ):
        super().__init__()

        # Set latent replay block
        assert (
            -1 <= latent_replay_block < num_blocks
        ), "Invalid latent replay block selection."
        self.latent_replay_block = latent_replay_block

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    tgt_len=tgt_len,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    device=device,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, get_activation=False):
        is_pattern, x = x

        if is_pattern:
            blocks_to_run = self.blocks
        else:
            assert self.latent_replay_block > -1, "Latent replay block not set."
            blocks_to_run = self.blocks[self.latent_replay_block :]

        activation = None
        for i, block in enumerate(blocks_to_run):
            if get_activation and i == self.latent_replay_block:
                activation = x.clone().detach()
            x = block(x)

        if get_activation:
            return x, activation
        else:
            return x

    def get_model_graph_information(self, x, get_activation=False):
        assert get_activation is False, "Get activation not implemented for model graph information."

        all_nodes = dict()
        ordered_nodes = list()

        is_pattern, x = x

        if is_pattern:
            blocks_to_run = self.blocks
        else:
            assert self.latent_replay_block > -1, "Latent replay block not set."
            blocks_to_run = self.blocks[self.latent_replay_block :]

        activation = None
        for i, block in enumerate(blocks_to_run):
            if get_activation and i == self.latent_replay_block:
                activation = x.clone().detach()
            x, block_nodes, block_ordered_nodes = block.get_model_graph_information(x, name="transformer_blocks_" + str(i))

            all_nodes = {**all_nodes, **block_nodes}
            ordered_nodes += block_ordered_nodes

        if get_activation:
            return x, activation, all_nodes, ordered_nodes
        else:
            return x, all_nodes, ordered_nodes

import torch.nn as nn
from torch.nn import functional as F


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x

    def get_model_graph_information(self, x, name):
        all_nodes = dict()
        ordered_nodes = []

        h = self.fc1(x)
        all_nodes[name + "_fc1"] = {
            "input": name[:-5] + "_norm2_output_data",
            "input_shape": tuple(x.shape[-2:]),
            "weight_shape": tuple(self.fc1.weight.shape),
            "bias_shape": tuple(self.fc1.bias.shape),
            "output_shape": tuple(h.shape[-2:]),
        }
        ordered_nodes.append(name + "_fc1")

        h = F.gelu(h)
        all_nodes[name + "_gelu"] = {
            "output_shape": tuple(h.shape[1:]),
        }
        ordered_nodes.append(name + "_gelu")

        x = self.fc2(h)
        all_nodes[name + "_fc2"] = {
            "input": name + "_gelu_output_data",
            "input_shape": tuple(h.shape[-2:]),
            "weight_shape": tuple(self.fc2.weight.shape),
            "bias_shape": tuple(self.fc2.bias.shape),
            "output_shape": tuple(x.shape[-2:]),
        }
        ordered_nodes.append(name + "_fc2")

        return x, all_nodes, ordered_nodes

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

        previous_shape = x.shape[-2:]
        h = self.fc1(x)
        all_nodes[name + "_fc1"] = {
            "input_a": name[:-5] + "_norm2_output_data",
            "input_b": (name + "_fc1_weight").upper(),
            "input_a_shape": tuple(previous_shape),
            "bias_shape": tuple(self.fc1.bias.shape),
            "output_shape": tuple(h.shape[-2:]),
        }
        ordered_nodes.append(name + "_fc1")

        h = F.gelu(h, approximate="tanh")
        all_nodes[name + "_gelu"] = {
            "shape": tuple(h.shape[1:]),
            "input": name + "_fc1_output_data",
        }
        ordered_nodes.append(name + "_gelu")

        x = self.fc2(h)
        all_nodes[name + "_fc2"] = {
            "input_a": name + "_gelu_output_data",
            "input_b": (name + "_fc2_weight").upper(),
            "input_a_shape": tuple(h.shape[-2:]),
            "bias_shape": tuple(self.fc2.bias.shape),
            "output_shape": tuple(x.shape[-2:]),
        }
        ordered_nodes.append(name + "_fc2")

        return x, all_nodes, ordered_nodes

import torch
from torch import nn
from torch.nn import functional as F


class DQN(nn.Module):
    def __init__(self, num_actions, device):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(in_features=8, out_features=256)

        self.hidden_layer_1 = nn.Linear(in_features=256, out_features=256)
        # self.hidden_layer_2 = nn.Linear(in_features=256, out_features=128)
        # self.hidden_layer_3 = nn.Linear(in_features=16, out_features=8)

        self.output_layer = nn.Linear(in_features=256, out_features=num_actions)

        self.device = device

    def forward(self, x):
        x = x.to(self.device)

        x = self.input_layer(x)
        x = F.relu(x)

        x = self.hidden_layer_1(x)
        x = F.relu(x)
        # x = self.hidden_layer_2(x)
        # x = F.relu(x)
        # x = self.hidden_layer_3(x)
        # x = F.relu(x)

        x = self.output_layer(x)

        return x

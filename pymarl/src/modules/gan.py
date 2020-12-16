import torch
from torch import nn

class GeneratorMLP(nn.Module):

    def __init__(self, noise_size, hidden_size, data_size):
        super(GeneratorMLP, self).__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(noise_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, data_size),
        )

    def forward(self, x):
        # noise = torch.randn((len(x), self.hidden_size)).cuda()
        return self.layers(x.cuda())


class DiscriminatorMLP(nn.Module):

    def __init__(self, hidden_size, data_size):
        super(DiscriminatorMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(data_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x.cuda())  # outputs logits


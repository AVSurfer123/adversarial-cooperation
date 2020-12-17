import torch
from torch import nn

class GeneratorMLP(nn.Module):

    def __init__(self, noise_size, hidden_size, data_size):
        super(GeneratorMLP, self).__init__()
        self.hidden_size = hidden_size
        self.noise_size = noise_size + 1
        self.layers = nn.Sequential(
            nn.Linear(self.noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, data_size),
        )

    def generate_noise(self, batch_size):
        return torch.randn(batch_size, self.noise_size)

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


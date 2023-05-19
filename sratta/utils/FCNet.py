import torch
import torch.nn.functional as F


class FCNet(torch.nn.Module):
    def __init__(self, dim, num_hidden_neurons, output_dim):
        super(FCNet, self).__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.num_hidden_neurons = num_hidden_neurons
        self.linear_1 = torch.nn.Linear(dim, self.num_hidden_neurons, bias=True)
        self.linear_2 = torch.nn.Linear(
            self.num_hidden_neurons, self.output_dim, bias=True
        )

    def forward(self, x):
        size_batch = x.shape[0]
        result = self.linear_2(F.relu(self.linear_1(x.reshape(size_batch, -1))))
        if self.output_dim == 1:
            return result[..., 0]
        else:
            return result

import torch
import torch.nn as nn
import numpy as np

# Model
# MultiLayerPerceptron
MIN = torch.Tensor(np.array([0,0]))
MAX = torch.Tensor(np.array([4,9]))

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(2, 200, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
             )
        self.linear_out = nn.Linear(200,4)
        self.linear_out.weight = nn.Parameter(torch.zeros(4,200))
        self.linear_out.bias = nn.Parameter(torch.zeros(4))

    def forward(self, x):
        x = (x - MIN)/(MAX-MIN)
        x = self.fc_layers(x)
        x = self.linear_out(x)
        return x
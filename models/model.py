import torch
from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden: list = (100, 10), convex=False) -> None:
        super().__init__()

        self.convex = convex

        assert len(hidden) == 2

        self.fc1 = nn.Linear(in_features, hidden[0])
        self.fc2 = nn.Linear(self.fc1.out_features, hidden[1])
        self.fc3 = nn.Linear(self.fc2.out_features, out_features)

        self.W1y = nn.Linear(in_features, self.fc2.out_features, False)
        self.W2y = nn.Linear(in_features, out_features, False)

        self.double()

    def forward(self, x):
        x_in = x.clone()

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x) + self.W1y(x_in)
        x = F.relu(x)
        x = self.fc3(x) + self.W2y(x_in)

        x = F.sigmoid(x)

        return x

    def step(self):
        if not self.convex:
            raise NotImplementedError("This network is not created convex.")
        with torch.no_grad():
            F.relu(self.W1y.weight.data, inplace=True)
            F.relu(self.W2y.weight.data, inplace=True)

from torch import nn
from torch.nn import functional as F


class CentralizedModel(nn.Module):
    def __init__(self):
        super(CentralizedModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

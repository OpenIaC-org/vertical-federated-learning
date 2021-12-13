from torch import nn, optim


class LabelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LabelNet()
        self.image_client_output = None
        self.grad_from_label = None
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9)

    def forward(self, image_client_output):
        self.image_client_output = image_client_output
        outputs = self.model(image_client_output)
        return outputs

    def backward(self):
        self.grad_from_label = self.image_client_output.grad.clone()
        return self.grad_from_label

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def step(self):
        self.optimizer.step()


class LabelNet(nn.Module):
    def __init__(self):
        super(LabelNet, self).__init__()
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        return self.fc3(x)

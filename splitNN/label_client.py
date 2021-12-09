from torch import nn, optim


class LabelClient(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LabelNet()
        self.image_client_output = None
        self.grad_from_label = None

    def forward(self, image_client_output):
        self.image_client_output = image_client_output
        outputs = self.model(image_client_output)
        return outputs

    def server_backward(self):
        self.grad_from_label = self.image_client_output.grad.clone()
        return self.grad_from_label

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def get_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)


class LabelNet(nn.Module):
    def __init__(self):
        super(LabelNet, self).__init__()
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        return self.fc3(x)

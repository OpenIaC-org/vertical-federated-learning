import torch


class SplitNN(torch.nn.Module):
    def __init__(self, image_client, label_client):
        super().__init__()
        self.image_client = image_client
        self.label_client = label_client
        self.image_optimizer = image_client.get_optimizer()
        self.label_optimizer = label_client.get_optimizer()

        self.image_client_output = None

    def forward(self, inputs):
        self.image_client_output = self.image_client(inputs)
        outputs = self.label_client(self.image_client_output)
        return outputs

    def backward(self):
        grad_to_image_client = self.label_client.backward()
        self.image_client.backward(grad_to_image_client)

    def zero_grads(self):
        self.image_optimizer.zero_grad()
        self.label_optimizer.zero_grad()

    def step(self):
        self.image_optimizer.step()
        self.label_optimizer.step()

    def train(self):
        self.image_client.train()
        self.label_client.train()

    def eval(self):
        self.image_client.eval()
        self.label_client.eval()

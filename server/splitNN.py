import torch


class SplitNN(torch.nn.Module):
    def __init__(self, image_client, label_client):
        super().__init__()
        self.image_client = image_client
        self.label_client = label_client

    def forward(self):
        image_client_output = self.image_client.forward()
        return self.label_client.forward(image_client_output)

    def backward(self):
        grad_to_image_client = self.label_client.backward()
        self.image_client.backward(grad_to_image_client)

    def zero_grads(self):
        self.image_client.zero_grads()
        self.label_client.zero_grads()

    def step(self):
        self.image_client.step()
        self.label_client.step()

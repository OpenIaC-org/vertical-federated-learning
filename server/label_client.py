from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from label_model import LabelModel
from utils import *

BATCH_SIZE = 128

transform = Compose([ToTensor(),
                     Normalize((0.5,), (0.5,)),
                     ])


def accuracy(label, output):
    pred = output.argmax(dim=1, keepdim=True)
    return pred.eq(label.view_as(pred)).sum().item() / pred.shape[0]


class LabelClient():
    def __init__(self):
        self.model = LabelModel()
        self.criterion = nn.CrossEntropyLoss()

        self.trainset = MNIST('./data', download=True, transform=transform)
        self.trainloader = DataLoader(self.trainset, batch_size=BATCH_SIZE)
        self.loader_iterator = iter(self.trainloader)
        self.max_batch_count = len(self.trainset) // BATCH_SIZE
        self.batch_counter = 0

    def forward(self, image_client_output):
        if self.max_batch_count == self.batch_counter:
            self.loader_iterator = iter(self.trainloader)
            self.batch_counter = 0
        _, labels = next(self.loader_iterator)
        output = self.model.forward(image_client_output)
        current_loss = self.criterion(output, labels)
        current_loss.backward()
        current_accuracy = accuracy(labels, output)
        self.batch_counter += 1
        return current_loss, current_accuracy

    def predict(self, image_client_output):
        output = self.model.forward(image_client_output)
        return output.argmax(dim=1, keepdim=True).item()

    def backward(self):
        return self.model.backward()

    def zero_grads(self):
        self.model.zero_grad()

    def step(self):
        self.model.step()

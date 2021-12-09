from torch import nn
import torch
from torch.utils.data.dataset import TensorDataset
from torchvision.transforms.transforms import ToTensor
from image_client import ImageClient
from label_client import LabelClient
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from splitNN import SplitNN


image_client = ImageClient()
label_client = LabelClient()

splitnn = SplitNN(image_client, label_client)


def accuracy(label, output):
    pred = output.argmax(dim=1, keepdim=True)
    return pred.eq(label.view_as(pred)).sum().item() / pred.shape[0]


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
trainset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)
train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    epoch_loss = 0
    epoch_outputs = []
    epoch_labels = []
    for i, data in enumerate(train_dataloader):
        splitnn.zero_grads()
        inputs, labels = data

        outputs = splitnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        epoch_loss += loss.item() / len(train_dataloader.dataset)

        epoch_outputs.append(outputs)
        epoch_labels.append(labels)

        splitnn.backward()
        splitnn.step()

    print(epoch_loss, accuracy(torch.cat(epoch_labels),
                               torch.cat(epoch_outputs)))

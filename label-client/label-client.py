from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from utils import *

labels = preprocess(MNIST('./data', download=True, transform=ToTensor()))
print(len(labels))

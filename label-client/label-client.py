from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from utils import *

data = MNIST('./data', download=True, transform=ToTensor())
data = remove_image(data)
print(data[0])

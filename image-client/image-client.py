from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


from utils import *

images = preprocess(MNIST('./data', download=True, transform=ToTensor()))
print(len(images))

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from utils import *

class ImageClient:
  def __init__(self):
    self.images = preprocess(MNIST('./data', download=True, transform=ToTensor()))
  
  async def handler(self):
    self.websocket = await create_websocket()
    while True:
      msg = await self.websocket.recv()

client = ImageClient()
asyncio.get_event_loop().run_until_complete(client.handler())
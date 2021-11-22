from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from utils import *

class LabelClient:
  def __init__(self):
    self.labels = preprocess(MNIST('./data', download=True, transform=ToTensor()))
  
  async def handler(self):
    self.websocket = await create_websocket()
    while True:
      msg = await self.websocket.recv()

client = LabelClient()
asyncio.get_event_loop().run_until_complete(client.handler())

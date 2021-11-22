from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pickle

from utils import *

class LabelClient:
  def __init__(self):
    self.labels = preprocess(MNIST('./data', download=True, transform=ToTensor()))
  
  async def handler(self):
    self.websocket = await create_websocket()
    while True:
      msg = await self.websocket.recv()
      if msg == 'get_ids':
        await self.websocket.send('ids')
        await self.websocket.send(pickle.dumps([lab[1] for lab in self.labels]))
      
      if msg == 'common_ids':
        self.ids = await self.websocket.recv()
        print(self.ids[:10])


client = LabelClient()
asyncio.get_event_loop().run_until_complete(client.handler())

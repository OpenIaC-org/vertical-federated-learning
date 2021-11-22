from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pickle

from utils import *

class ImageClient:
  def __init__(self):
    self.images = preprocess(MNIST('./data', download=True, transform=ToTensor()))
  
  async def handler(self):
    self.websocket = await create_websocket()
    while True:
      msg = await self.websocket.recv()
      if msg == 'get_ids':
        await self.websocket.send('ids')
        await self.websocket.send(pickle.dumps([img[1] for img in self.images]))
      
      if msg == 'common_ids':
        self.ids = pickle.loads(await self.websocket.recv())

client = ImageClient()
asyncio.get_event_loop().run_until_complete(client.handler())
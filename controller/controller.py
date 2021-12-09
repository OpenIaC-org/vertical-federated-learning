import websockets
import asyncio
from aioconsole import ainput

class Controller:
  def __init__(self):
    self.actions = {  
      'get ids': 'Get list of unique IDs from clients',
      'permute': 'Permute the list of IDs, so that each client is in sync',
    }

  async def handler(self):
    while True:
      message = await ainput()
      await self.websocket.send(message)

  async def connect(self):
    self.websocket = await websockets.connect('ws://localhost:8000', ping_interval=None)
    print('Connected. Type your messages and send with enter. Possible actions are:')
    for key, val in self.actions.items():
      print(f'  - {key}: {val}')
    await self.handler()

controller = Controller()
asyncio.get_event_loop().run_until_complete(controller.connect())

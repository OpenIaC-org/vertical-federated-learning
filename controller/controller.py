import websockets
import asyncio
from aioconsole import ainput

class Controller:
  async def handler(self):
    while True:
      message = await ainput()
      print(message)
      await self.websocket.send(message)

  async def connect(self):
    self.websocket = await websockets.connect('ws://localhost:8000', ping_interval=None)
    print('Connected. Type your messages and send with enter.')
    await self.handler()

controller = Controller()
asyncio.get_event_loop().run_until_complete(controller.connect())

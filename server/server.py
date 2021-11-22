import websockets
import asyncio

class Server:
  def __init__(self):
    self.clients = {}

  async def handler(self, websocket, path):
    while True:
      try:
        data = await websocket.recv()
        if data == 'connect':
          await self.handle_connection(websocket)
      except websockets.exceptions.ConnectionClosed:
        break
  
  async def handle_connection(self, websocket):
    self.clients[str(websocket.id)] = websocket
    await websocket.send(str(websocket.id))
    print('Client ' + str(websocket.id) + ' connected')

server = Server()
start_server = websockets.serve(server.handler, '', 8000, ping_timeout=None)

try:
  asyncio.get_event_loop().run_until_complete(start_server)
  print('Server started on port 8000')
  asyncio.get_event_loop().run_forever()
except KeyboardInterrupt:
  print('Server stopped by keyboard')
import websockets
import asyncio
import pickle
from aioconsole import ainput

class Server:
  def __init__(self):
    self.clients = {}
    self.ids = []

  async def handler(self, websocket, path):
    while True:
      try:
        data = await websocket.recv()

        await self.client_endpoints(websocket, data)
        await self.controller_endpoints(websocket, data)

      except websockets.exceptions.ConnectionClosed:
        break

  async def client_endpoints(self, websocket, data):
    if data == 'connect':
      await self.handle_connection(websocket)

    if data == 'ids':
      ids = pickle.loads(await websocket.recv())
      self.ids.append(ids)
      if len(self.ids) == 2:
        print('Got all IDs')

  async def controller_endpoints(self, websocket, data):
    if data == 'get ids':
      await self.get_ids()
    
    if data == 'permute':
      common_ids = await self.find_id_permutation()
      await self.send_to_all('common_ids')
      await self.send_to_all(pickle.dumps(common_ids))
      print('Sent common IDs to all clients')


  async def handle_connection(self, websocket):
    self.clients[str(websocket.id)] = websocket
    await websocket.send(str(websocket.id))
    print('Client ' + str(websocket.id) + ' connected')

  async def get_ids(self):
    if len(self.clients) != 2:
      print('Incorrect number of clients: ' + str(len(self.clients)))
      return
    
    await self.send_to_all('get_ids')
  
  async def find_id_permutation(self):
    if len(self.ids) != 2:
      print('Incorrect number of ids: ' + str(len(self.ids)))
      return
    
    print('Finding common ids')
    common_ids = list(set(self.ids[0]).intersection(self.ids[1]))
    print(f'Found {len(common_ids)} common ids')
    return common_ids
  
  async def send_to_all(self, data):
    for client in self.clients:
      await self.clients[client].send(data)

server = Server()
start_server = websockets.serve(server.handler, '', 8000, ping_timeout=None)

try:
  asyncio.get_event_loop().run_until_complete(start_server)
  print('Server started on port 8000')
  asyncio.get_event_loop().run_forever()
except KeyboardInterrupt:
  print('Server stopped by keyboard')
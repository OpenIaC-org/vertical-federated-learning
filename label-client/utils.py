import random
import asyncio
import websockets

def preprocess(data):
  """ Preprocesses data for label client
  1. Removes all image data from the data
  2. Adds ids to the data
  3. Shuffles the data
  4. Removes a random number of labels
  """
  print('Preprocessing labels')
  data = [x[1] for x in data]
  data = [(x, i) for i, x in enumerate(data)]
  random.shuffle(data)
  for _ in range(random.randint(len(data) * 0.05, len(data) * 0.1)):
    data.pop()
  return data

async def create_websocket():
  try:
    websocket = await websockets.connect('ws://localhost:8000', ping_interval=None)
    await websocket.send('connect')
    print('Connected to server')
    return websocket
  except:
    print('Connection failed, trying again in 2 seconds')
    await asyncio.sleep(2)
    return await create_websocket()
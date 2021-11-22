import asyncio
import random
import matplotlib.pyplot as plt
import websockets

def preprocess(data):
  """ Preprocesses data for image client
  1. Removes all labels from the data
  2. Adds ids to the data
  3. Shuffles the data
  4. Removes a random number of images
  """
  print('Preprocessing images')
  data = [x[0] for x in data]
  data = [(x, i) for i, x in enumerate(data)]
  random.shuffle(data)
  for _ in range(random.randint(len(data) * 0.05, len(data) * 0.1)):
    data.pop()
  return data

def show_image(image, multiple=False):
  plt.title(f'id: {image[1]}')
  plt.imshow(image[0].numpy().squeeze())
  if not multiple:
    plt.show()

def show_images(images):
  for i, image in enumerate(images):
    plt.subplot(1, len(images), i+1)
    plt.axis('off')
    show_image(image, multiple=True)
  plt.show()

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
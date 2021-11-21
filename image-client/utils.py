import random
import matplotlib.pyplot as plt

def preprocess(data):
  """ Preprocesses data for image client
  1. Removes all labels from the data
  2. Adds ids to the data
  3. Shuffles the data
  """
  print('Preprocessing images')
  data = [x[0] for x in data]
  data = [(x, i) for i, x in enumerate(data)]
  random.shuffle(data)
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
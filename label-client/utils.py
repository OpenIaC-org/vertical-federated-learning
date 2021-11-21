import random

def preprocess(data):
  """ Preprocesses data for label client
  1. Removes all image data from the data
  2. Adds ids to the data
  3. Shuffles the data
  """
  print('Preprocessing labels')
  data = [x[1] for x in data]
  data = [(x, i) for i, x in enumerate(data)]
  random.shuffle(data)
  return data
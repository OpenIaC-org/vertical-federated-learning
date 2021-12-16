import pickle
from flask.helpers import send_file
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from image_model import ImageModel
from flask import Flask, request
from flask_cors import CORS, cross_origin

from utils import *
import logging

app = Flask(__name__)
cors = CORS(app)
log = logging.getLogger('werkzeug')
log.disabled = True

PORT = 5000
SERVER_PORT = 8000
BATCH_SIZE = 128

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

trainset = MNIST('./data', download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128)
loader_iterator = iter(trainloader)
model = ImageModel()
connect_to_server(SERVER_PORT, PORT)
max_batch_count = len(trainset) // BATCH_SIZE
batch_counter = 0


@app.post('/forward')
def forward():
    global model, batch_counter, loader_iterator
    if max_batch_count == batch_counter:
        loader_iterator = iter(trainloader)
        batch_counter = 0
    images, _ = next(loader_iterator)
    batch_counter += 1
    return pickle.dumps(model.forward(images))


@app.post('/backward')
def backward():
    global model
    gradient = pickle.loads(request.data)
    model.backward(gradient)
    return 'Hopefully I learned something!'


@app.get('/zero_grads')
def zero_grads():
    global model
    model.zero_grad()
    return 'Zeroed grads'


@app.get('/step')
def step():
    global model
    model.step()
    return 'Stepped'


@cross_origin()
@app.get('/metadata')
def metadata():
    return send_file('metadata.json')


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=PORT)

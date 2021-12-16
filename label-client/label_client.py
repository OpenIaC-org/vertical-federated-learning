import pickle
from flask.helpers import send_file
from torch import nn
from torch.nn.functional import batch_norm
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from label_model import LabelModel
from flask import Flask, request
from flask_cors import CORS, cross_origin

from utils import *
import logging

app = Flask(__name__)
cors = CORS(app)
log = logging.getLogger('werkzeug')
log.disabled = True

PORT = 5001
SERVER_PORT = 8000
BATCH_SIZE = 128

transform = Compose([ToTensor(),
                     Normalize((0.5,), (0.5,)),
                     ])


connect_to_server(SERVER_PORT, PORT)
model = LabelModel()
criterion = nn.CrossEntropyLoss()

trainset = MNIST('./data', download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE)
loader_iterator = iter(trainloader)
max_batch_count = len(trainset) // BATCH_SIZE
batch_counter = 0


def accuracy(label, output):
    pred = output.argmax(dim=1, keepdim=True)
    return pred.eq(label.view_as(pred)).sum().item() / pred.shape[0]


@app.post('/forward')
def forward():
    global model, loss, batch_counter, loader_iterator
    if max_batch_count == batch_counter:
        loader_iterator = iter(trainloader)
        batch_counter = 0
    image_client_output = pickle.loads(request.data)
    _, labels = next(loader_iterator)
    output = model.forward(image_client_output)
    current_loss = criterion(output, labels)
    current_loss.backward()
    current_accuracy = accuracy(labels, output)
    batch_counter += 1
    return pickle.dumps((output, current_loss, current_accuracy))


@app.post('/backward')
def backward():
    global model
    return pickle.dumps(model.backward())


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

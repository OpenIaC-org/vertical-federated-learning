import pickle
from flask.json import jsonify
from torch import nn
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from label_model import LabelModel
from flask import Flask, request, jsonify

from utils import *

app = Flask(__name__)

PORT = 5001
SERVER_PORT = 8000

labels = preprocess(MNIST('./data', download=True, transform=ToTensor()))
connect_to_server(SERVER_PORT, PORT)
model = LabelModel()
criterion = nn.CrossEntropyLoss()


@app.get('/ids')
def get_ids():
    return jsonify([img[1] for img in labels])


@app.post('/ids')
def post_ids():
    global labels
    print('Reordering labels')

    common_ids = request.json
    labels = reorder_labels(common_ids, labels)
    print(common_ids[:10])
    return 'Reordered labels'


def loss(outputs, batch):
    global model, labels, criterion
    labs = []
    for id in batch:
        label = next(
            (lab[0] for lab in labels if lab[1] == id), None)
        if label is None:
            raise Exception('Label not found')
        labs.append(label)
    labs = torch.stack(labs)
    return criterion(outputs, labs), accuracy(labs, outputs)


def accuracy(label, output):
    pred = output.argmax(dim=1, keepdim=True)
    return pred.eq(label.view_as(pred)).sum().item() / pred.shape[0]


@app.post('/forward')
def forward():
    global model, loss
    data = pickle.loads(request.data)
    image_client_output, batch = pickle.loads(
        data['image_client_output']), data['batch']
    output = model.forward(image_client_output)
    current_loss, current_accuracy = loss(output, batch)
    return pickle.dumps((output, current_loss, current_accuracy))


@app.post('/backward')
def backward():
    global model
    return model.backward()


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


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=PORT)

import pickle
from torchvision import transforms
from torchvision.datasets import MNIST
from image_model import ImageModel
from flask import Flask, request, jsonify
import torch

from utils import *

app = Flask(__name__)

PORT = 5000
SERVER_PORT = 8000

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

images = preprocess(MNIST('./data', download=True, transform=transform))
model = ImageModel()
connect_to_server(SERVER_PORT, PORT)


@app.get('/ids')
def get_ids():
    return jsonify([img[1] for img in images])


@app.post('/ids')
def post_ids():
    global images
    print('Reordering images')

    common_ids = request.json
    images = reorder_images(common_ids, images)
    print(common_ids[:10])
    return 'Reordered images'


@app.post('/forward')
def forward():
    global model, images
    ids = request.json
    img = []
    for id in ids:
        image = next(
            (img[0] for img in images if img[1] == id), None)
        if image is None:
            raise Exception('Image not found')
        img.append(image)
    img = torch.stack(img)
    img = img.float()

    return pickle.dumps(model.forward(img))


@app.post('/backward')
def backward():
    global model
    gradient = request.json
    return pickle.dumps(model.backward(gradient))


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

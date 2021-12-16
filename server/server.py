from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin
import numpy as np
from client_connection import ImageClientConnection
from label_client import LabelClient

from splitNN import SplitNN
from utils import *
import time

app = Flask(__name__)
cors = CORS(app)

NUMBER_OF_IMAGES = 13500
BATCH_SIZE = 128


image_client = None
label_client = LabelClient()
splitNN = None


def initialize_splitNN():
    global splitNN, image_client, label_client

    if splitNN is None and image_client is not None:
        print('Initializing splitNN')
        splitNN = SplitNN(image_client, label_client)


@app.get('/connect-image-client')
def image_connect():
    global image_client

    client_ip = request.access_route[-1]
    port = request.args.get('port')
    image_client = ImageClientConnection(port, client_ip)
    print(f'Image client connected on {client_ip}:{port}')
    return 'Connected'


@app.get('/train')
def train():
    if splitNN is None:
        initialize_splitNN()

    print('Training')
    start = time.time()
    for epoch in range(1, 11):
        batches = NUMBER_OF_IMAGES // BATCH_SIZE
        epoch_loss = 0
        epoch_accuracy = []
        for _ in range(batches):
            splitNN.zero_grads()
            loss, accuracy = splitNN.forward()
            epoch_accuracy.append(accuracy)
            epoch_loss += (loss.item() / (NUMBER_OF_IMAGES / BATCH_SIZE))
            splitNN.backward()
            splitNN.step()
        print(f'Epoch {epoch}: {epoch_loss} - {np.mean(epoch_accuracy)}')
    print(f'Training time: {time.time() - start}')
    return 'Done training!'


@cross_origin()
@app.get('/metadata')
def metadata():
    return send_file('metadata.json')


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)

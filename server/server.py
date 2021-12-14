import random
from flask import Flask, request
import numpy as np
from client_connection import ImageClientConnection, LabelClientConnection

from splitNN import SplitNN
from utils import *

# import logging
# logging.basicConfig(filename='./logs/log.log', level=logging.DEBUG)

app = Flask(__name__)

image_client = None
label_client = None
splitNN = None
ids = []


def get_ids():
    global image_client, label_client, ids

    ids.append(image_client.get_ids())
    ids.append(label_client.get_ids())

    print('Got all IDs')


def find_common_ids():
    common_ids = find_id_permutation()
    image_client.send_ids(common_ids),
    label_client.send_ids(common_ids)


def initialize_splitNN():
    global splitNN, image_client, label_client

    if splitNN is None and label_client is not None and image_client is not None:
        print('Initializing splitNN')
        splitNN = SplitNN(image_client, label_client)


def find_id_permutation():
    global ids

    if len(ids) != 2:
        print('Incorrect number of ids: ' + str(len(ids)))
        return

    print('Finding common ids')
    common_ids = list(set(ids[0]).intersection(ids[1]))
    print(f'Found {len(common_ids)} common ids')
    ids = common_ids
    return common_ids


@app.get('/connect-image-client')
def image_connect():
    global image_client

    client_ip = request.access_route[-1]
    port = request.args.get('port')
    image_client = ImageClientConnection(port, client_ip)
    print(f'Image client connected on {client_ip}:{port}')
    return 'Connected'


@app.get('/connect-label-client')
def label_connect():
    global label_client

    client_ip = request.access_route[-1]
    port = request.args.get('port')
    label_client = LabelClientConnection(port, client_ip)
    print(f'Label client connected on {client_ip}:{port}')
    return 'Connected'


@app.get('/format-ids')
def format_ids():
    initialize_splitNN()
    get_ids()
    find_common_ids()
    return 'Formatted ids'


@app.get('/train')
def train():
    if splitNN is None:
        print('SplitNN not initialized')
        return 'SplitNN not initialized'

    print('Training')
    for epoch in range(1, 11):
        current_ids = random.sample(ids, len(
            ids) // 2)  # Random order
        epoch_loss = 0
        chunk_size = 64
        current_chunks = chunks(current_ids, chunk_size)
        epoch_accuracy = []
        for batch in current_chunks:
            splitNN.zero_grads()
            output, loss, accuracy = splitNN.forward(batch)
            epoch_accuracy.append(accuracy)
            # loss.backward()
            epoch_loss += (loss.item() / (len(current_ids) / chunk_size))
            splitNN.backward()
            splitNN.step()
        print(f'Epoch {epoch}: {epoch_loss} - {np.mean(epoch_accuracy)}')
    return 'Done training!'


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)

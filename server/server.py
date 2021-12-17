from flask import Flask, request, send_file, Response
from flask_cors import CORS, cross_origin
import numpy as np
from numpy.core.numeric import cross
from client_connection import ImageClientConnection
from label_client import LabelClient
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import io

from splitNN import SplitNN
from utils import *
import time

app = Flask(__name__)
cors = CORS(app)

NUMBER_OF_IMAGES = 2000
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
@app.get('/predict/<image_id>')
def predict(image_id):
    data = label_client.trainset[int(image_id)]
    label = data[1]
    prediction = splitNN.predict(int(image_id))
    return {'label': label, 'prediction': prediction}


@cross_origin()
@app.get('/image/<image_id>')
def get_image(image_id):
    data = label_client.trainset[int(image_id)]
    image = data[0]
    plot = create_plot(image)
    return Response(plot.getvalue(), mimetype='image/png')


def create_plot(image):
    img = image.numpy().squeeze()
    sizes = np.shape(img)
    fig = Figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward=False)
    ax.imshow(img)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return output


@cross_origin()
@app.get('/metadata')
def metadata():
    return send_file('metadata.json')


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)

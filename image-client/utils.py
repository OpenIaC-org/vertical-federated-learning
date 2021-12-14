import random
import matplotlib.pyplot as plt
import requests
import time


def preprocess(data):
    """ Preprocesses data for image client
    1. Removes all labels from the data
    2. Adds ids to the data
    3. Shuffles the data
    4. Removes a random number of images
    """
    print('Preprocessing images')
    data = data.data[:5000]
    data = [(x, i) for i, x in enumerate(data)]
    # random.shuffle(data)
    # for _ in range(random.randint(len(data) * 0.05, len(data) * 0.1)):
    #     data.pop()
    return data


def show_image(image, multiple=False):
    if len(image) == 2:
        plt.title(f'id: {image[1]}')
        plt.imshow(image[0].numpy().squeeze())
    else:
        plt.imshow(image.numpy().squeeze())
    if not multiple:
        plt.show()


def show_images(images):
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.axis('off')
        show_image(image, multiple=True)
    plt.show()


def connect_to_server(server_port, port):
    requests.get(
        f'http://127.0.0.1:{server_port}/connect-image-client?port={port}')


def reorder_images(ids, images):
    output = []
    for id in ids:
        image = next(
            (img for img in images if img[1] == id), None)
        if image is not None:
            output.append(image)
    return output

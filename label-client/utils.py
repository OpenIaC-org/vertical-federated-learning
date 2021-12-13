import random
import requests


def preprocess(data):
    """ Preprocesses data for label client
    1. Removes all image data from the data
    2. Adds ids to the data
    3. Shuffles the data
    4. Removes a random number of labels
    """
    print('Preprocessing labels')
    data = data.targets[:5000]
    data = [(x, i) for i, x in enumerate(data)]
    random.shuffle(data)
    for _ in range(random.randint(len(data) * 0.05, len(data) * 0.1)):
        data.pop()
    return data


def connect_to_server(server_port, port):
    requests.get(
        f'http://localhost:{server_port}/connect-label-client?port={port}')


def reorder_labels(ids, labels):
    output = []
    for id in ids:
        label = next(
            (lab for lab in labels if lab[1] == id), None)
        if label is not None:
            output.append(label)
    return output

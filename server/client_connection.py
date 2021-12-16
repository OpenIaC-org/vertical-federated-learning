import pickle
import requests


class ClientConnection():
    def __init__(self, port, host):
        self.host = host
        self.port = port

    def zero_grads(self):
        requests.get(f'http://{self.host}:{self.port}/zero_grads')

    def step(self):
        requests.get(f'http://{self.host}:{self.port}/step')


class ImageClientConnection(ClientConnection):
    def __init__(self, port, host):
        super().__init__(port, host)

    def forward(self):
        return requests.post(f'http://{self.host}:{self.port}/forward').content

    def backward(self, gradient):
        gradient = pickle.dumps(gradient)
        requests.post(
            f'http://{self.host}:{self.port}/backward', data=gradient)

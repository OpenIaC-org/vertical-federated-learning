import pickle
import requests


class ClientConnection():
    def __init__(self, port, host):
        self.host = host
        self.port = port

    def get_ids(self):
        return requests.get(f'http://{self.host}:{self.port}/ids').json()

    def send_ids(self, common_ids):
        print('Sending ids to clients')
        return requests.post(f'http://{self.host}:{self.port}/ids', json=common_ids)

    def zero_grads(self):
        requests.get(f'http://{self.host}:{self.port}/zero_grads')

    def step(self):
        requests.get(f'http://{self.host}:{self.port}/step')


class ImageClientConnection(ClientConnection):
    def __init__(self, port, host):
        super().__init__(port, host)

    def forward(self, input):
        return requests.post(f'http://{self.host}:{self.port}/forward', json=input).content

    def backward(self, gradient):
        gradient = pickle.dumps(gradient)
        requests.post(
            f'http://{self.host}:{self.port}/backward', data=gradient)


class LabelClientConnection(ClientConnection):
    def __init__(self, port, host):
        super().__init__(port, host)

    def loss(self, outputs, ids):
        return requests.post(f'http://{self.host}:{self.port}/loss', json={'outputs': outputs, 'ids': ids}).json()

    def forward(self, image_output, batch):
        params = {'image_client_output': image_output, 'batch': batch}
        data = requests.post(
            f'http://{self.host}:{self.port}/forward', data=pickle.dumps(params)).content
        data = pickle.loads(data)
        return data

    def backward(self):
        grad = requests.post(
            f'http://{self.host}:{self.port}/backward').content
        return pickle.loads(grad)

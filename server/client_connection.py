import pickle
import requests
from torch.nn.functional import batch_norm


class ClientConnection():
    def __init__(self, port):
        self.port = port

    def get_ids(self):
        return requests.get('http://localhost:' + str(self.port) + '/ids').json()

    def send_ids(self, common_ids):
        print('Sending ids to clients')
        return requests.post('http://localhost:' + str(self.port) +
                             '/ids', json=common_ids)

    def forward(self, input):
        return requests.post(f'http://localhost:{self.port}/forward', json=input).content

    def backward(self, gradient=None):
        return requests.post(f'http://localhost:{self.port}/backward', data=gradient).content

    def zero_grads(self):
        requests.get(f'http://localhost:{self.port}/zero_grads')

    def step(self):
        requests.get(f'http://localhost:{self.port}/step')


class LabelClientConnection(ClientConnection):
    def __init__(self, websocket):
        super().__init__(websocket)

    def loss(self, outputs, ids):
        return requests.post(f'http://localhost:{self.port}/loss', json={'outputs': outputs, 'ids': ids}).json()

    def forward(self, image_output, batch):
        params = {'image_client_output': image_output, 'batch': batch}
        data = requests.post(
            f'http://localhost:{self.port}/forward', data=pickle.dumps(params)).content
        data = pickle.loads(data)
        return data

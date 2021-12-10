import random
import websockets
import asyncio
import pickle

from splitNN import SplitNN
from utils import *

import logging
logging.basicConfig(filename='./logs/log.log', level=logging.DEBUG)


class Server:
    def __init__(self):
        self.image_client = None
        self.label_client = None
        self.splitNN = None
        self.ids = []

    async def handler(self, websocket, path):
        while True:
            try:
                logging.info("Handler -> Receive", exc_info=True)
                data = await websocket.recv()
                logging.info("Handler -> Return", exc_info=True)

                await self.client_endpoints(websocket, data)
                await self.controller_endpoints(websocket, data)

            except websockets.exceptions.ConnectionClosed:
                break

    async def client_endpoints(self, websocket, data):
        if data == 'label_connect':
            self.label_client = websocket
            print('Label client connected')

        if data == 'image_connect':
            self.image_client = websocket
            print('Image client connected')

        if self.splitNN is None and self.label_client is not None and self.image_client is not None:
            print('Initializing splitNN')
            self.splitNN = SplitNN(self.image_client, self.label_client)

        if data == 'ids':
            logging.info("IDs -> Receive", exc_info=True)
            ids = pickle.loads(await websocket.recv())
            logging.info("IDs -> Return", exc_info=True)
            self.ids.append(ids)
            if len(self.ids) == 2:
                print('Got all IDs')

    async def controller_endpoints(self, websocket, data):
        if data == 'get ids':
            await self.get_ids()

        if data == 'permute':
            self.common_ids = await self.find_id_permutation()
            await self.send_to_both('common_ids')
            await self.send_to_both(pickle.dumps(self.common_ids))
            print('Sent common IDs to all clients')

        if data == 'train':
            if self.splitNN is None:
                print('SplitNN not initialized')
                return
            print('Starting training')
            await self.train()

    async def get_ids(self):
        if self.label_client is None:
            print('Label client not connected')
            return

        if self.image_client is None:
            print('Image client not connected')
            return

        await self.send_to_both('get_ids')

    async def find_id_permutation(self):
        if len(self.ids) != 2:
            print('Incorrect number of ids: ' + str(len(self.ids)))
            return

        print('Finding common ids')
        common_ids = list(set(self.ids[0]).intersection(self.ids[1]))
        print(f'Found {len(common_ids)} common ids')
        return common_ids

    async def send_to_both(self, data):
        await self.image_client.send(data)
        await self.label_client.send(data)

    async def train(self):
        ids = random.sample(self.common_ids, len(
            self.common_ids))  # Random order
        loss = 0
        current_chunks = chunks(ids, 64)
        for batch in current_chunks:
            print(batch)
            await self.splitNN.zero_grads()
            outputs = await self.splitNN.forward(batch)
            loss = await self.splitNN.loss(outputs, batch)
            loss.backward()
            loss += loss.item() / len(current_chunks)
            await self.splitNN.backward()
            await self.splitNN.step()
        print(loss)


server = Server()
start_server = websockets.serve(server.handler, '', 8000, ping_timeout=None)

try:
    asyncio.get_event_loop().run_until_complete(start_server)
    print('Server started on port 8000')
    asyncio.get_event_loop().run_forever()
except KeyboardInterrupt:
    print('Server stopped by keyboard')

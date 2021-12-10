import pickle
import logging


class ClientConnection():
    def __init__(self, websocket):
        self.websocket = websocket

    async def forward(self, inputs):
        await self.websocket.send('forward')
        await self.websocket.send(pickle.dumps(inputs))
        logging.info("Forward -> Receive", exc_info=True)
        output = await self.websocket.recv()
        logging.info("Forward -> Return", exc_info=True)
        return pickle.loads(output)

    async def backward(self, gradient=None):
        await self.websocket.send('backward')
        if gradient is None:
            logging.info("Backward -> Receive", exc_info=True)
            output = await self.websocket.recv()
            logging.info("Backward -> Return", exc_info=True)
            return pickle.loads(output)
        else:
            await self.websocket.send(pickle.dumps(gradient))

    async def zero_grads(self):
        await self.websocket.send('zero_grads')

    async def step(self):
        await self.websocket.send('step')


class LabelClientConnection(ClientConnection):
    def __init__(self, websocket):
        super().__init__(websocket)

    async def loss(self, outputs, ids):
        await self.websocket.send('loss')
        await self.websocket.send(pickle.dumps(outputs))
        await self.websocket.send(pickle.dumps(ids))
        logging.info("Forward -> Receive", exc_info=True)
        output = await self.websocket.recv()
        logging.info("Forward -> Return", exc_info=True)
        return pickle.loads(output)

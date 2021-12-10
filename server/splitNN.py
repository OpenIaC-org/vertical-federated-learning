import torch

from client_connection import ClientConnection


class SplitNN(torch.nn.Module):
    def __init__(self, image_websocket, label_websocket):
        super().__init__()
        self.image_client = ClientConnection(image_websocket)
        self.label_client = ClientConnection(label_websocket)

    async def forward(self, input_ids):
        image_client_output = await self.image_client.forward(input_ids)
        return await self.label_client.forward(image_client_output)

    async def backward(self):
        grad_to_image_client = await self.label_client.backward()
        await self.image_client.backward(grad_to_image_client)

    async def zero_grads(self):
        await self.image_client.zero_grads()
        await self.label_client.zero_grads()

    async def step(self):
        await self.image_client.step()
        await self.label_client.step()

    async def loss(self, outputs, batch):
        return await self.label_client.loss(outputs, batch)

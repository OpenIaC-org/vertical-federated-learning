import requests

SERVER_PORT = 8000

print(f'Run commands on the server on port {SERVER_PORT}')
print('Possible commands:')
print('\t- "train" start federated training')

while True:
    command = input()
    requests.get(f'http://localhost:{SERVER_PORT}/{command}')

# vfl
Vertical Federated Learning implemented with Docker

## To run:
1. Ensure you have docker and docker-compose installed
2. Navigate to the root folder
3. `docker-compose build`
4. `docker-compose up`
5. Navigate to controller/web-controller
6. Ensure you have npm installed and run `npm start`
7. Get the IP and port from the server in the format `IP:PORT` and put it into the web-controller text field (Example: `127.0.0.1:8000`)
8. Press train and observe the training output in the console where the docker-compose was started, or in the Docker Desktop output

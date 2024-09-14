import torch
import random
from flwr.common import NumPyClient
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


# Helper functions for getting/setting specific layer parameters
def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    net.load_state_dict(state_dict, strict=True)

def get_layer_weights(net, layers):
    return {name: param.cpu().numpy() for name, param in net.named_parameters() if name in layers}

def update_layer_weights(net, layer_weights):
    for name, param in net.named_parameters():
        if name in layer_weights:
            param.data = torch.tensor(layer_weights[name]).to(param.device)

def set_trainable_layers(net, layers_to_train):
    for name, param in net.named_parameters():
        param.requires_grad = name in layers_to_train

def train(model, trainloader, criterion, optimizer, layers_to_train, epochs = 1,  device="cpu"):
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = name in layers_to_train  # Train only specific layers
    model.to(device)
    for i in range(epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def test(model, valloader, criterion, device="cpu"):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in valloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(valloader.dataset)
    return total_loss / len(valloader), accuracy
"""what do i want my client to do 
1 get weights and list of lauers to train 
train the layers of the model 
send the layes to the model """
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, global_model, device="cpu"):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.global_model = global_model
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.layers_to_send = []  # No layers initially

    def get_parameters(self, config):
        # Send specific layers from the previous round
        if self.layers_to_send:
            return get_layer_weights(self.global_model, self.layers_to_send)
        else:
            return get_parameters(self.global_model)

    def fit(self, parameters, config):
        # Receive layers to train from the global server
        layers_to_train_from_server = config.get("layers_to_train", [])

        # Set the parameters received from the server for the specific layers
        if parameters:
            update_layer_weights(self.net, parameters)

        # Set trainable layers as instructed by the server
        set_trainable_layers(self.net, layers_to_train_from_server)

        # Train the model on the client's dataset with only the specified layers
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=0.001)
        train(self.net, self.trainloader, self.criterion, optimizer, layers_to_train_from_server, device=self.device)
        
        print(f"Client finished training with layers: {layers_to_train_from_server}")

        # Collect updated weights for the trained layers
        updated_weights = get_layer_weights(self.net, layers_to_train_from_server)
        self.layers_to_send = layers_to_train_from_server  # Set layers to send for the next round

        return updated_weights, len(self.trainloader), {}

    def evaluate(self, parameters, config):
        # Set parameters before evaluating
        if parameters:
            update_layer_weights(self.net, parameters)

        loss, accuracy = test(self.net, self.valloader, self.criterion, device=self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

# Example usage of FlowerClient
def main():
    # Assuming `net`, `trainloader`, `valloader`, and `global_model` are predefined
    client = FlowerClient(net, trainloader, valloader, global_model, device="cuda" if torch.cuda.is_available() else "cpu")
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()

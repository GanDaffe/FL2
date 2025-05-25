import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from algorithm import *
from utils import *
from models import CNN, BN_CNN

EXP_NAME = 'FedAvg'
NUM_DOMAINS = 3
NUM_CLIENTS_PER_DOMAIN = 3
BATCH_SIZE = 32
RANDOM_STATE = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

c1_data = pd.read_feather('/data/Domain 1.feather')
c2_data = pd.read_feather('/data/Domain 2.feather')
c3_data = pd.read_feather('/data/Domain 3.feather')

data_full = [c1_data, c2_data , c3_data]

clients_dataset = get_clients_dataset(data_full, NUM_DOMAINS, NUM_CLIENTS_PER_DOMAIN)

train_set, validation_set = [], []

for i in range(len(clients_dataset)):
    train, val = train_test_split(clients_dataset[i], test_size=0.2, random_state=RANDOM_STATE)
    train_set.append(train)
    validation_set.append(val)

trainloaders = [DataLoader(train_set[i], batch_size=BATCH_SIZE) for i in range(len(train_set))]
valloaders = [DataLoader(validation_set[i], batch_size=BATCH_SIZE) for i in range(len(validation_set))]

NUM_ROUNDS = 3
LEARNING_RATE = 0.01

net = BN_CNN(in_channel=1, num_classes=3)
criterion = nn.CrossEntropyLoss()

def base_client_fn(cid: str):
    idx = int(cid)
    return BaseClient(cid, net, trainloaders[idx], valloaders[idx], criterion).to_client()

current_parameters = ndarrays_to_parameters(get_parameters(net))
client_resources = {"num_cpus": 1, "num_gpus": 0.2} if DEVICE == "cuda" else {"num_cpus": 1, "num_gpus": 0.0}

fl.simulation.start_simulation(
            client_fn           = base_client_fn,
            num_clients         = NUM_DOMAINS * NUM_CLIENTS_PER_DOMAIN,
            config              = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy            = FedAvg(
                learning_rate       = LEARNING_RATE,
                exp_name            = EXP_NAME,
                algo_name           = 'FedAvg',
                net                 = net,
                device              = DEVICE,
                num_rounds          = NUM_ROUNDS,
                num_clients         = NUM_DOMAINS * NUM_CLIENTS_PER_DOMAIN,
                current_parameters  = current_parameters,
                ),
            client_resources     = client_resources
        )
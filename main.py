import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Context

from algorithm import *
from utils import *
from models import CNN, BN_CNN

ALGO_NAME = 'fedbn'
EXP_NAME = '_1epcs'
NUM_DOMAINS = 3
NUM_CLIENTS_PER_DOMAIN = 3
NUM_EPOCHS = 1
BATCH_SIZE = 32
RANDOM_STATE = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(RANDOM_STATE)

c1_data = pd.read_feather('/home/tzt5387/Documents/Hiệu Projects/FL2/data/Domain 1.feather')
c2_data = pd.read_feather('/home/tzt5387/Documents/Hiệu Projects/FL2/data/Domain 2.feather')
c3_data = pd.read_feather('/home/tzt5387/Documents/Hiệu Projects/FL2/data/Domain 3.feather')

data_full = [c1_data, c2_data , c3_data]
num_samples = sum([len(x) for x in data_full])
clients_dataset, client_domain_mapper = get_clients_dataset(data_full, NUM_DOMAINS, NUM_CLIENTS_PER_DOMAIN)

train_set, validation_set = [], []

for i in range(len(clients_dataset)):
    train, val = train_test_split(clients_dataset[i], test_size=0.2, random_state=RANDOM_STATE)
    train_set.append(train)
    validation_set.append(val)

trainloaders = [DataLoader(train_set[i], batch_size=BATCH_SIZE) for i in range(len(train_set))]
valloaders = [DataLoader(validation_set[i], batch_size=BATCH_SIZE) for i in range(len(validation_set))]

NUM_ROUNDS = 50
LEARNING_RATE = 0.003

algo = None

if ALGO_NAME == 'fedavg': 
    algo = FedAvg
elif ALGO_NAME == 'fedyogi': 
    algo = FedYogi   
elif ALGO_NAME == 'fedprox': 
    algo = FedProx 
    client = FedProxClient
elif ALGO_NAME == 'fedadp': 
    algo = FedAdp
elif ALGO_NAME == 'fedadam': 
    algo = FedAdam 
elif ALGO_NAME == 'fedavgM':
    algo = FedAvgM
elif ALGO_NAME == 'fedadagrad': 
    algo = FedAdagrad 
elif ALGO_NAME == 'moon': 
    algo = MOON
    client = MoonClient
elif ALGO_NAME == 'fednova': 
    algo = FedNovaStrategy
elif ALGO_NAME == 'scaffold': 
    algo = SCAFFOLD
elif ALGO_NAME == 'fedbn':
    algo = FedBN 
    
def base_client_fn(context: Context):
    cid = int(context.node_config["partition-id"])
    idx = cid
    criterion = nn.CrossEntropyLoss()
    net = BN_CNN(in_channel=1, num_classes=3)
    if ALGO_NAME == 'fednova': 
        client_dataset_ratio: float = int(num_samples / (NUM_DOMAINS * NUM_CLIENTS_PER_DOMAIN)) / num_samples
        return FedNovaClient(cid, client_domain_mapper[idx], net, trainloaders[idx], valloaders[idx], criterion, num_epochs=NUM_EPOCHS, ratio=client_dataset_ratio).to_client()
    elif ALGO_NAME == 'moon':  
        net_moon = init_model() 
        return MoonClient(cid, client_domain_mapper[idx], net_moon, trainloaders[idx], valloaders[idx], criterion, num_epochs=NUM_EPOCHS, dir='moon_cp/moon_models').to_client()
    elif ALGO_NAME == 'scaffold': 
        c_local = load_c_local(idx)
        return SCAFFOLD_CLIENT(cid, client_domain_mapper[idx], net, trainloaders[idx], valloaders[idx], criterion, num_epochs=NUM_EPOCHS, c_local=c_local).to_client()  
    elif ALGO_NAME == 'fedprox': 
        return FedProxClient(cid, client_domain_mapper[idx], net, trainloaders[idx], valloaders[idx], criterion, num_epochs=NUM_EPOCHS).to_client()
    elif ALGO_NAME == 'fedbn': 
        return FedBNClient(cid, client_domain_mapper[idx], net, trainloaders[idx], valloaders[idx], criterion, num_epochs=NUM_EPOCHS).to_client()
    return BaseClient(cid, client_domain_mapper[idx], net, trainloaders[idx], valloaders[idx], criterion, num_epochs=NUM_EPOCHS).to_client()

net_ = init_model() if ALGO_NAME == 'moon' else BN_CNN(in_channel=1, num_classes=3)     
current_parameters = ndarrays_to_parameters(get_parameters(net_))
client_resources = {"num_cpus": 2, "num_gpus": 0.2} if DEVICE == "cuda" else {"num_cpus": 1, "num_gpus": 0.0}

def get_strategy(): 
    if ALGO_NAME == 'scaffold': 
        return algo(
                learning_rate       = LEARNING_RATE,
                exp_name            = EXP_NAME,
                algo_name           = ALGO_NAME,
                device              = DEVICE,
                net                 = net_,
                num_rounds          = NUM_ROUNDS,
                num_clients         = NUM_DOMAINS * NUM_CLIENTS_PER_DOMAIN,
                current_parameters  = current_parameters,
            )
    elif ALGO_NAME == 'fednova':
        state_dict = net_.state_dict()
        learnable_params = {
            name: param for name, param in state_dict.items()
            if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name
        }
        init_parameters = ndarrays_to_parameters([param.cpu().numpy() for param in learnable_params.values()])
        return algo(
                learning_rate       = LEARNING_RATE,
                exp_name            = EXP_NAME,
                algo_name           = ALGO_NAME,
                device              = DEVICE,
                num_rounds          = NUM_ROUNDS,
                num_clients         = NUM_DOMAINS * NUM_CLIENTS_PER_DOMAIN,
                current_parameters  = init_parameters,
            )
    else:
        return algo(
                learning_rate       = LEARNING_RATE,
                exp_name            = EXP_NAME,
                algo_name           = ALGO_NAME,
                device              = DEVICE,
                num_rounds          = NUM_ROUNDS,
                num_clients         = NUM_DOMAINS * NUM_CLIENTS_PER_DOMAIN,
                current_parameters  = current_parameters,
            )
        
fl.simulation.start_simulation(
            client_fn           = base_client_fn,
            num_clients         = NUM_DOMAINS * NUM_CLIENTS_PER_DOMAIN,
            config              = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy            = get_strategy(),
            client_resources     = client_resources
        )
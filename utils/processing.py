from torch.utils.data import DataLoader
import numpy as np
import random
import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, targets, transform=None):
        self.data = [input_data[i].unsqueeze(0) for i in range(input_data.size(0))]
        self.targets = targets
        self.classes = torch.unique(targets).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"Seeds set to {seed_value}")

def domain_partition(X, y, num_clients):
    num_classes = np.unique(y).shape[0]
    class_indices = [[] for _ in range(num_classes)]

    for i, lab in enumerate(y):
        class_indices[lab].append(i)

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        proportions = np.random.dirichlet(np.ones(num_clients) * 5)
        indices = np.array(class_indices[c])
        np.random.shuffle(indices)

        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)

        for i, idx in enumerate(split_indices):
            client_indices[i].extend(idx.tolist())

    client_data = [(torch.from_numpy(X[client_idx]), torch.from_numpy(y[client_idx])) for client_idx in client_indices]

    return client_data

def data_processing(df, NUM_FEATURES):
   y_train = df['Label']
   flow_id = df['flow_id']

   df = df/255

   X_train = df.drop(['Label', 'flow_id'], axis=1)
   X_train = X_train.to_numpy()

   X_train = X_train.reshape(-1, 20, NUM_FEATURES)
   y_train = y_train.to_numpy()

   y_train = y_train.reshape(-1,20)[:,-1]
   return X_train, y_train

def get_clients_dataset(full_domain_data, num_domains, num_clients_per_domain):   
    set_seed(42)
    all_data = [] 

    for domain in full_domain_data:
        all_data.append(data_processing(domain, 256))

    domain_clients = []
    for data, label in all_data:  
        domain_clients.extend(domain_partition(data, label, num_clients_per_domain))

    for i in range(len(domain_clients)):
        domain_clients[i] = CustomDataset(domain_clients[i][0], domain_clients[i][1])
    return domain_clients
    


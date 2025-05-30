from algorithm.base.client import BaseClient
from utils import set_parameters, train
import torch

class FedProxClient(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        optimizer = torch.optim.SGD(params=self.net.parameters(), lr=config['learning_rate'])
        loss, accuracy = train(self.net, self.trainloader, self.criterion, optimizer, device=config['device'], num_epochs=self.num_epochs, proximal_mu=config["proximal_mu"])
        return self.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid}

from algorithm.base.client import BaseClient 
from algorithm.import_lib import *
from algorithm.fednova.fednova_utils import ProxSGD
from torch import nn

class FedNovaClient(BaseClient):
    def __init__(self, *args, ratio, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio = ratio
        self.optimizer = ProxSGD(
            params=self.net.parameters(),
            lr=1,
            ratio=self.ratio
        )
        
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        params = [
            val["cum_grad"].cpu().numpy()
            for _, val in self.optimizer.state_dict()["state"].items()
        ]
        return params

    def set_parameters(self, parameters):
        for param, new_param in zip(self.net.parameters(), parameters):
            param.data.copy_(torch.tensor(new_param).to(param.device))
        self.optimizer.set_model_params(parameters)


    def fit(self, parameters, config):

        lr = config['learning_rate'] 
        self.set_parameters(parameters)
        self.optimizer.set_lr(lr)

        criterion = nn.CrossEntropyLoss()
        train_loss, train_acc = train(
            self.net,
            self.trainloader,
            criterion,
            self.optimizer,
            config['device'],
            self.num_epochs,
        )

        grad_scaling_factor = self.optimizer.get_gradient_scaling()

        metrics = {
            "accuracy": train_acc,
            "loss": train_loss,
            "tau": grad_scaling_factor["tau"],     
            "local_norm": grad_scaling_factor["local_norm"],
            "weight": grad_scaling_factor["weight"]
        }

        return self.get_parameters({}), len(self.trainloader.sampler), metrics

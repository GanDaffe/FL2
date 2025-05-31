from algorithm.base.client import BaseClient 
from algorithm.import_lib import *
from algorithm.fednova.fednova_utils import ProxSGD

class FedNovaClient(BaseClient):
    def __init__(self, *args, ratio, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio = ratio
        
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        params = [
            val["cum_grad"].cpu().numpy()
            for _, val in self.optimizer.state_dict()["state"].items()
        ]
        return params

    def set_parameters(self, parameters, lr):
        state_dict = {name: torch.tensor(param) for name, param in zip(self.net.state_dict().keys(), parameters)}
        self.net.load_state_dict(state_dict)

        self.optimizer = ProxSGD(
            params=self.net.parameters(),
            lr=lr,
            ratio=self.ratio
        )
  
        for group in self.optimizer.param_groups:
            for p in group['params']:
                self.optimizer.state[p]['old_init'] = p.data.clone()

    def fit(self, parameters, config):

        lr = config['learning_rate'] 

        self.set_parameters(parameters, lr)

        train_loss, train_acc = train(
            self.net,
            self.optimizer,
            self.trainloader,
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

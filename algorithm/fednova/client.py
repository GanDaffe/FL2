from algorithm.base.client import BaseClient 
from algorithm.import_lib import *
from algorithm.fednova.fednova_utils import ProxSGD
from torch import nn

class FedNovaClient(BaseClient):
    def __init__(self, *args, ratio, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio = ratio
        
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        params = []

        for name, param in self.net.named_parameters():
            state = self.optimizer.state.get(param, {})
            cum_grad = state.get("cum_grad", None)
            if cum_grad is not None:

                if cum_grad.shape == param.shape:
                    params.append(cum_grad.cpu().numpy())
                else:
                    print(f"[Warning] Skipping cum_grad for {name} due to shape mismatch: cum_grad {cum_grad.shape} vs param {param.shape}")
            else:
                print(f"[Info] No cum_grad found for {name}")
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

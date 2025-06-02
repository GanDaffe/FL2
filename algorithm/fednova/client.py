from algorithm.base.client import BaseClient 
from algorithm.import_lib import *
from algorithm.fednova.fednova_utils import ProxSGD
from torch import nn

class FedNovaClient(BaseClient):
    def __init__(self, *args, ratio, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio = ratio
        
    def get_parameters(self, config: Dict[str, Scalar], optimizer) -> NDArrays:
        params = [
            val["cum_grad"].cpu().numpy()
            for _, val in optimizer.state_dict()["state"].items()
            if "cum_grad" in val
        ]
        return params

    def set_parameters(self, parameters):
        state_dict = self.net.state_dict()
        param_names = [
            name for name in state_dict.keys()
            if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name
        ]
        params_dict = zip(param_names, parameters)
        new_state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        state_dict.update(new_state_dict)
        self.net.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):

        lr = config['learning_rate'] 
        self.set_parameters(parameters)

        optimizer = ProxSGD(
            params=self.net.parameters(),
            lr=lr,
            ratio=self.ratio
        )

        criterion = nn.CrossEntropyLoss()
        train_loss, train_acc = train(
            self.net,
            self.trainloader,
            criterion,
            optimizer,
            config['device'],
            self.num_epochs,
        )

        grad_scaling_factor = optimizer.get_gradient_scaling()

        metrics = {
            "accuracy": train_acc,
            "loss": train_loss,
            "tau": grad_scaling_factor["tau"],     
            "local_norm": grad_scaling_factor["local_norm"],
            "weight": grad_scaling_factor["weight"]
        }

        return self.get_parameters({}, optimizer), len(self.trainloader.sampler), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc, prec, rec, f1, TP, FP, FN, TN = test(self.net, self.valloader, config["device"])
        return loss, len(self.valloader.sampler), {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1, 
            "TP": TP, 
            "FP": FP, 
            "FN": FN, 
            "TN": TN
        }
from logging import INFO, log
from algorithm.import_lib import *
from algorithm.base.strategy import FedAvg

class FedNovaStrategy(FedAvg):
    def __init__(self, gmf = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = self.learning_rate
        self.gmf = gmf
        self.global_momentum_buffer = []
        if self.current_parameters is not None:
            state_dict = self.current_parameters  
            self.global_parameters = [
                param.cpu().numpy() for name, param in state_dict.items()
                if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name
            ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:

        local_tau = [fit_res.metrics["tau"] for _, fit_res in results]
        tau_eff = np.sum(local_tau)

        aggregate_parameters = []
        
        for _, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            scale = tau_eff / float(fit_res.metrics["local_norm"])
            scale *= float(fit_res.metrics["weight"])

            aggregate_parameters.append((params, scale))


        agg_cum_gradient = aggregate(aggregate_parameters)        
        self.update_server_params(agg_cum_gradient)

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        examples = [fit_res.num_examples for _, fit_res in results]
        loss = sum(losses) / sum(examples)
        accuracy = sum(corrects) / sum(examples)

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)
        print(f"train_loss: {loss} - train_acc: {accuracy}")


        return ndarrays_to_parameters(self.global_parameters), {}

    def update_server_params(self, cum_grad: List[np.ndarray]):
        for i, layer_cum_grad in enumerate(cum_grad):
            global_param = self.global_parameters[i]
            global_shape = global_param.shape
            grad_shape = layer_cum_grad.shape

            print(f"\n[Update Param] Layer {i}")
            print(f"  Global Param Shape: {global_shape} | Gradient Shape: {grad_shape}")
            print(f"  Global Param dtype: {global_param.dtype} | Grad dtype: {layer_cum_grad.dtype}")
            print(f"  Global Param min: {global_param.min():.6f}, max: {global_param.max():.6f}")
            print(f"  Gradient min: {layer_cum_grad.min():.6f}, max: {layer_cum_grad.max():.6f}")

            if global_shape != grad_shape:
                print(f"[ERROR] Shape mismatch at layer {i}: global shape {global_shape}, grad shape {grad_shape}")
                continue  

            if self.gmf != 0:
                if len(self.global_momentum_buffer) <= i:
                    buf = layer_cum_grad / self.lr
                    self.global_momentum_buffer.append(buf)
                    print(f"  [Init Momentum] Buffer initialized for Layer {i}")
                else:
                    self.global_momentum_buffer[i] *= self.gmf
                    self.global_momentum_buffer[i] += layer_cum_grad / self.lr
                    print(f"  [Momentum Update] gmf={self.gmf}, lr={self.lr:.6f}")
                self.global_parameters[i] -= self.global_momentum_buffer[i] * self.lr
                print(f"  [Param Updated] with momentum")
            else:
                self.global_parameters[i] -= layer_cum_grad
                print(f"  [Param Updated] without momentum")

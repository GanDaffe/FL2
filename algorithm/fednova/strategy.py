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
            self.global_parameters = parameters_to_ndarrays(
                self.current_parameters
            )

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

    def update_server_params(self, cum_grad: NDArrays):
        print(f"Sample global param: {self.global_parameters[i].flatten()[:5]}")
        print(f"Sample gradient: {layer_cum_grad.flatten()[:5]}")

        for i, layer_cum_grad in enumerate(cum_grad):
            global_shape = self.global_parameters[i].shape
            grad_shape = layer_cum_grad.shape
            print(f"[Update Param] Layer {i} | Global Param Shape: {global_shape} | Gradient Shape: {grad_shape}")

            if self.gmf != 0:
                if len(self.global_momentum_buffer) < len(cum_grad):
                    buf = layer_cum_grad / self.lr
                    self.global_momentum_buffer.append(buf)
                else:
                    self.global_momentum_buffer[i] *= self.gmf
                    self.global_momentum_buffer[i] += layer_cum_grad / self.lr

                try:
                    self.global_parameters[i] -= self.global_momentum_buffer[i] * self.lr
                except ValueError as e:
                    print(f"[ERROR] Broadcast error at layer {i}: {e}")
                    raise

            else:
                try:
                    self.global_parameters[i] -= layer_cum_grad
                except ValueError as e:
                    print(f"[ERROR] Broadcast error at layer {i}: {e}")
                    raise

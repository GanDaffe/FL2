from algorithm.import_lib import *
from algorithm.base.strategy import FedAvg

class FedProx(FedAvg):
    def __init__(
        self,
        *args,
        proximal_mu: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.proximal_mu = proximal_mu

    def __repr__(self) -> str:
        return "FedProx"


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        config = {"learning_rate": self.learning_rate, "proximal_mu": self.proximal_mu}
        self.learning_rate *= self.decay_rate
        fit_ins = FitIns(parameters, config)

        fit_configs = [(client, fit_ins) for client in clients]
        return fit_configs


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        self.current_parameters = ndarrays_to_parameters(
            aggregate([(parameters_to_ndarrays(f.parameters), f.num_examples) for _, f in results])
        )
        examples = [f.num_examples for _, f in results]
        total = sum(examples)

        def weighted_avg(metric_name):
            return sum(f.num_examples * f.metrics[metric_name] for _, f in results) / total

        loss = weighted_avg("loss")
        acc = weighted_avg("accuracy")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(acc)

        print(f"Train R{server_round}: loss={loss:.4f}, acc={acc:.4f}")

        return self.current_parameters, {}


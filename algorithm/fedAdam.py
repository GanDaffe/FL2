from algorithm.import_lib import *
from algorithm.base.strategy import FedAvg

class FedAdam(FedAvg): 
    
    def __init__(
            self, 
            *args, 
            eta: float = 1e-2, 
            tau: float = 1e-3, 
            beta_1: float = 0.0, 
            beta_2: float = 0.0, 
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.eta = eta 
        self.tau = tau 
        self.beta_1 = beta_1 
        self.beta_2 = beta_2 

        self.m_t: Optional[NDArrays] = None
        self.v_t: Optional[NDArrays] = None

    def __repr__(self) -> str:
        return "FedAdam"
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
        self.current_parameters = parameters_to_ndarrays(self.current_parameters)
        fedavg_weights_aggregate = aggregate(weights_results)
        # Adam
        delta_t: NDArrays = [
            x - y for x, y in zip(fedavg_weights_aggregate, self.current_parameters)
        ]

        # m_t
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            np.multiply(self.beta_1, x) + (1 - self.beta_1) * y
            for x, y in zip(self.m_t, delta_t)
        ]

        # v_t
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [
            self.beta_2 * x + (1 - self.beta_2) * np.multiply(y, y)
            for x, y in zip(self.v_t, delta_t)
        ]

        new_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(self.current_parameters, self.m_t, self.v_t)
        ]

        self.current_parameters = ndarrays_to_parameters(new_weights)
        metrics_aggregated = {}
        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        examples = [fit_res.num_examples for _, fit_res in results]
        loss = sum(losses) / sum(examples)
        accuracy = sum(corrects) / sum(examples)

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        return self.current_parameters, metrics_aggregated

from algorithm.import_lib import *
from algorithm.base.strategy import FedAvg

class SCAFFOLD(FedAvg):

    def __init__(
        self,
        *args,
        net,
        **kwargs, 
    ) -> None:
        super().__init__(*args, **kwargs) 

        self.net = net 
        self.c_global = [torch.zeros_like(param) for param in net.parameters()]
        self.current_weights = [w.astype(np.float32) for w in parameters_to_ndarrays(self.current_parameters)]
        self.num_clients = self.num_clients
        self.global_learning_rate = self.learning_rate
    
    def initialize_parameters(self, client_manager):
        return ndarrays_to_parameters(self.current_weights)
    
    def __repr__(self) -> str:
        return 'SCAFFOLD'


    def configure_fit(
        self, server_round, parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(sample_size, min_num_clients)
       
        c_global = []
        for param in self.c_global:
            c_global += param.cpu().flatten().tolist() 
            
        global_c_numpy = np.array(c_global, dtype=np.float64)
        global_c_bytes = global_c_numpy.tobytes()
        config = {"learning_rate": self.global_learning_rate, "device": self.device, "c_global": global_c_bytes}
        self.global_learning_rate = self.global_learning_rate * self.decay_rate


        return [(client, FitIns(parameters, config)) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_ndarrays(parameters_aggregated)

        for current_weight, fed_weight in zip(self.current_weights, fedavg_weights_aggregate):
            current_weight += fed_weight * self.global_learning_rate
            
        c_delta_sum = [np.zeros_like(c_global.cpu()) for c_global in self.c_global]

        for _, fit_res in results:
            c_delta = np.frombuffer(fit_res.metrics["c_delta"], dtype=np.float64)
            for i in range(len(c_delta_sum)):
                c_delta_sum[i] += np.array(c_delta[i], dtype=np.float64)

        for i in range(len(self.c_global)):
            c_delta_avg = c_delta_sum[i] / self.num_clients
            self.c_global[i] += torch.tensor(c_delta_avg, device=self.c_global[i].device)
            
        return ndarrays_to_parameters(self.current_weights), {}
    
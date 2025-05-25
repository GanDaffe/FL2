from algorithm.import_lib import *

class FedAvg(fl.server.strategy.Strategy):
    def __init__(
        self, exp_name, algo_name, num_rounds, num_clients, device,
        decay_rate=0.995, fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
        learning_rate=0.01, current_parameters=None
    ):
        super().__init__()
        self.exp_name = exp_name
        self.algo_name = algo_name
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.learning_rate = learning_rate
        self.current_parameters = current_parameters
        self.device = device
        self.decay_rate = decay_rate
        self.result = {
            "round": [],
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1": []
        }

    def __repr__(self):
        return "FedAvg"

    def initialize_parameters(self, client_manager):
        return self.current_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(sample_size, min_num_clients)
        config = {"learning_rate": self.learning_rate, "device": self.device}
        self.learning_rate *= self.decay_rate
        return [(client, FitIns(parameters, config)) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
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

    def configure_evaluate(self, server_round, parameters, client_manager):
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(sample_size, min_num_clients)
        config = {"device": self.device}
        return [(client, EvaluateIns(parameters, config)) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        examples = [r.num_examples for _, r in results]
        total = sum(examples)

        def weighted_avg(metric_name):
            return sum(r.num_examples * r.metrics[metric_name] for _, r in results) / total

        loss = sum(r.num_examples * r.loss for _, r in results) / total
        acc = weighted_avg("accuracy")
        prec = weighted_avg("precision")
        rec = weighted_avg("recall")
        f1 = weighted_avg("f1")

        if server_round != 0:
            self.result["test_loss"].append(loss)
            self.result["test_accuracy"].append(acc)
            self.result["test_precision"].append(prec)
            self.result["test_recall"].append(rec)
            self.result["test_f1"].append(f1)

        print(f"Test R{server_round}: loss={loss:.4f}, acc={acc:.4f}, prec={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")

        if server_round == self.num_rounds:
            pd.DataFrame(self.result).to_csv(f"result/{self.algo_name}_{self.exp_name}.csv", index=False)

        return loss, {}

    def evaluate(self, server_round, parameters):
        return None

    def num_fit_clients(self, num_available):
        return max(int(num_available * self.fraction_fit), self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available):
        return max(int(num_available * self.fraction_evaluate), self.min_evaluate_clients), self.min_available_clients

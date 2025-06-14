from algorithm.import_lib import *

class BaseClient(fl.client.NumPyClient):
    def __init__(self, cid, domain_id, net, trainloader, valloader, criterion, num_epochs=1):
        self.cid = cid
        self.domain_id = domain_id 
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = criterion
        self.num_epochs = num_epochs

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        optimizer = torch.optim.SGD(self.net.parameters(), lr=config["learning_rate"])
        loss, acc = train(self.net, self.trainloader, self.criterion, optimizer, device=config["device"], num_epochs=self.num_epochs)
        
        return self.get_parameters(config), len(self.trainloader.sampler), {
            "loss": loss,
            "accuracy": acc,
            "id": self.cid
        }

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, acc, prec, rec, f1, TP, FP, FN, TN = test(self.net, self.valloader, config["device"])
        return loss, len(self.valloader.sampler), {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "domain_id": self.domain_id,
            "f1": f1, 
            "TP": TP, 
            "FP": FP, 
            "FN": FN, 
            "TN": TN
        }
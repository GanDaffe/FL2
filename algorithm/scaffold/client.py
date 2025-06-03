from algorithm.base.client import BaseClient 
from utils import set_parameters
from algorithm.import_lib import *
from algorithm.scaffold.scaffold_utils import set_c_local
from logging import INFO, log

class SCAFFOLD_CLIENT(BaseClient):
    def __init__(self, *args, c_local, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_local = c_local

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        lr = config['learning_rate']
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

        results = self.train_scaffold(
            net=self.net,
            trainloader=self.trainloader,
            epochs=self.num_epochs,
            learning_rate=lr,
            device=config['device'],
            config=config,
            c_local=self.c_local,
            parameters=parameters
        )
        return self.get_parameters(self.net), len(self.trainloader.dataset), results
    
    def train_scaffold(self, net, trainloader, epochs, learning_rate, device, config, c_local, parameters):
        c_global_bytes = config['c_global']
        c_global_np = np.frombuffer(c_global_bytes, dtype=np.float64).copy() 
        
        c_global = []
        idx = 0
        for param in net.parameters():
            param_size = param.numel() 
            param_shape = param.shape  
            c_global_param = torch.from_numpy(c_global_np[idx:idx + param_size]).reshape(param_shape).to(device)
            c_global.append(c_global_param)
            idx += param_size

        global_weight = [param.detach().clone().to(device) for param in net.parameters()]
        if c_local is None:
            log(INFO, f"No cache found for c_local")
            c_local = [torch.zeros_like(param, device=device) for param in net.parameters()]      

        net.to(device)
        net.train()

        loss_avg, running_corrects, tot_sample = 0, 0, 0

        for _ in range(epochs):
            prebatch_params = [param.detach().clone() for param in net.parameters()]

            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = net(images)
                loss = self.criterion(outputs, labels)

                predicted = torch.argmax(outputs, dim=1)
                running_corrects += torch.sum(predicted == labels).item()
                loss_avg += loss.item()
                tot_sample += images.shape[0]

                loss.backward()
                self.optimizer.step()
                
                for param, y_i, c_l, c_g in zip(net.parameters(), prebatch_params, c_local, c_global):
                    if param.requires_grad:
                        param.grad.data = y_i - (learning_rate * (param.grad.data - c_l + c_g.to(device)))

            y_delta = [param.detach().clone() - gw for param, gw in zip(net.parameters(), global_weight)]

            coef = 1 / (epochs * learning_rate)
            c_plus = [
                c_l - c_g + coef * (param_g - param_l)
                for c_l, c_g, param_l, param_g in zip(c_local, c_global, net.parameters(), global_weight)
            ]

            for param, new_w in zip(net.parameters(), y_delta):
                param.data = new_w.clone().detach()

            c_delta = [cp - cl for cp, cl in zip(c_plus, c_local)]

            set_c_local(self.cid, c_plus)

            c_delta_list = []
            for param in c_delta:
                c_delta_list += param.flatten().tolist()
                
            c_delta_numpy = np.array(c_delta_list, dtype=np.float64)
            c_delta_bytes = c_delta_numpy.tobytes()

            results = {
                "loss": loss_avg / tot_sample,
                "accuracy": running_corrects / tot_sample,
                "c_delta": c_delta_bytes,
            }
            return results
        

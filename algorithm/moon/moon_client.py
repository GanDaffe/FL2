from algorithm.base.client import BaseClient 
from algorithm.import_lib import *
from algorithm.moon.moon_utils import *

class MoonClient(BaseClient):

    def __init__(self, *args, dir, **kwargs):

        super().__init__(*args, **kwargs)
        if dir == None:
            self.dir = f'client_{self.cid}'
        else:
            self.dir = f'{dir}_{self.cid}'
        
        self.initial_parameters = self.net

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:

        temperature = config["temperature"]
        learning_rate = config["learning_rate"]
      
        set_parameters(self.net, parameters)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            prev_net = copy.deepcopy(self.net)
        else:
            prev_net_path = os.path.join(self.dir, "prev_net.pt")
            if os.path.exists(prev_net_path):
                prev_net = copy.deepcopy(self.initial_parameters)
                prev_net.load_state_dict(torch.load(prev_net_path))
            else:
                prev_net = copy.deepcopy(self.net)


        global_net = copy.deepcopy(self.initial_parameters)
        global_net.load_state_dict(self.net.state_dict())


        _, loss, acc = train_moon(
                            self.net,
                            global_net,
                            prev_net,
                            self.trainloader,
                            learning_rate,
                            temperature,
                            epochs=self.num_epochs, 
                            device=config['device']
                     )

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            torch.save(self.net.state_dict(), os.path.join(self.dir, "prev_net.pt"))

        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False, 'loss': loss, 'accuracy': acc}

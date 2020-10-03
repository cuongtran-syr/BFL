# Federated training starting from here
from agent import *
import numpy as np


class BaseFL(object):
    def __init__(self, configs=None):
        default_configs = {
            'num_clients': 100,
            'T': 20,  # num outer epochs 30-40
            'B': 2,  # branch size of tree,
            'params': {},
            'device': 'cpu'
        }
        self.curr_model = Net()
        self.R = self.num_clients/2.0

        if configs is not None:
            default_configs.update(configs)
            for key, val in default_configs.items():
                setattr(self, key, val)

        for key, val in default_configs.items():
            # set property for BaseFL object based on dictionary key-values.
            setattr(self, key, val)

        self.clients = [Agent_CLF(self.params[i]) for i in range(self.num_clients)]
        self.logs = {'val_acc': []}

    def shuffle_clients(self):
        return np.random.permutation(self.num_clients)

    def set_weights(self, ref_model):
        """
        Set model
        """
        self.curr_model = Net()
        copy_model(ref_model, self.curr_model)

    def get_weights(self):
        """
        get model  weights
        """
        w_dict = {}
        for name, param in self.curr_model.named_parameters():
            w_dict[name] = copy.deepcopy(param)
        return w_dict


class ChainFL(BaseFL):
    # extend the base federated learning class for chain topology
    def __init__(self, configs=None):
        super().__init__(configs)
        self.B = 1  # branch factor = 1

    def train(self):
        curr_model = None
        for _ in range(self.T):
            shuffled_clts = super().shuffle_clients()
            for clt in shuffled_clts:
                if curr_model is not None:
                    self.clients[clt].set_weights(curr_model)
                self.clients[clt].train()
                curr_model = copy.deepcopy(self.clients[clt].model)

            curr_acc = eval(curr_model, self.test_loader, self.device)
            print(curr_acc)
            self.logs['val_acc'].append(curr_acc)


class TreeFL(BaseFL):
    def __init__(self, configs=None):
        super().__init__(configs)

        self.h = [h for h in range(self.num_clients) if (self.B ** h - 1) / (self.B - 1) >= self.num_clients][
            0]  # height of tree
        self.index_leaf = (self.B ** (self.h - 1) - 1) / (self.B - 1) + 1
        self.num_leaves = float(self.num_clients - self.index_leaf + 1)

    def train(self):

        for t in range(self.T):
            model_list = []

            shuffled_clts = super().shuffle_clients()
            print(shuffled_clts)
            for i, clt in enumerate(shuffled_clts):
                parent_index = int(np.floor(
                    (i - 1) / self.B))  # get parent index of clt, check my write up, parent of a node i, is [(i-1)/B]
                if parent_index >= 0:
                    parent_model = copy.deepcopy(self.clients[shuffled_clts[parent_index]].model)
                    self.clients[clt].set_weights(parent_model)
                else:
                    if t >= 1:
                        self.clients[clt].model = copy.deepcopy(curr_model)

                self.clients[clt].train()

                if i >= self.index_leaf:
                    model_list.append(dict(self.clients[clt].model.named_parameters()))

                    # aggregation step applied for leave nodes
            with torch.no_grad():
                global_params = {}
                for param in model_list[0]:
                    param_data = model_list[0][param].data
                    for model_state in model_list[1:]:
                        param_data += model_state[param].data
                    param_data /= len(model_list)
                    global_params[param] = param_data

            # self.set_weights(agg_model_dict)
            curr_model = Net()
            curr_model.load_state_dict(global_params)
            self.curr_model = copy.deepcopy(curr_model)
            curr_acc = eval(self.curr_model, self.test_loader, self.device)
            print(curr_acc)
            self.logs['val_acc'].append(curr_acc)


class RingFL(BaseFL):
    def __init__(self, configs=None):
        super().__init__(configs)

    def train(self):
        for t in range(self.T):
            model_list = []
            shuffled_clts = super().shuffle_clients()
            for clt in shuffled_clts:
                if t >= 1:
                    self.clients[clt].model = copy.deepcopy(curr_model)
                self.clients[clt].train()
                model_list.append(dict(self.clients[clt].model.named_parameters()))

            with torch.no_grad():
                global_params = {}
                for param in model_list[0]:
                    param_data = model_list[0][param].data
                    for model_state in model_list[1:]:
                        param_data += model_state[param].data
                    param_data /= len(model_list)
                    global_params[param] = param_data

            curr_model = Net()
            curr_model.load_state_dict(global_params)
            self.curr_model = copy.deepcopy(curr_model)
            curr_acc = eval(self.curr_model, self.test_loader, self.device)
            print(curr_acc)
            self.logs['val_acc'].append(curr_acc)



class FedAvg(BaseFL):
    def __init__(self, configs=None):
        super().__init__(configs)

    def train(self):
        for t in range(self.T):
            model_list = []
            shuffled_clts = super().shuffle_clients()
            for clt in shuffled_clts[:self.R]:
                if t >= 1:
                    self.clients[clt].model = copy.deepcopy(curr_model)
                self.clients[clt].train()
                model_list.append(dict(self.clients[clt].model.named_parameters()))

            with torch.no_grad():
                global_params = {}
                for param in model_list[0]:
                    param_data = model_list[0][param].data
                    for model_state in model_list[1:]:
                        param_data += model_state[param].data
                    param_data /= len(model_list)
                    global_params[param] = param_data

            curr_model = Net()
            curr_model.load_state_dict(global_params)
            self.curr_model = copy.deepcopy(curr_model)
            curr_acc = eval(self.curr_model, self.test_loader, self.device)
            print(curr_acc)
            self.logs['val_acc'].append(curr_acc)



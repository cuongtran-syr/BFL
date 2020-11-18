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


        if configs is not None:
            default_configs.update(configs)
            for key, val in default_configs.items():
                setattr(self, key, val)

        for key, val in default_configs.items():
            # set property for BaseFL object based on dictionary key-values.
            setattr(self, key, val)

        self.clients = [Agent_CLF(self.params[i]) for i in range(self.num_clients)]
        self.logs = {'val_acc': [], 'val_acc_iter':[]}

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

    def agg_model(self, model_list, start, end ):
        with torch.no_grad():
            global_params = {}
            for param in model_list[start]:
                param_data = model_list[start][param].data
                num_ = 1.0
                for model_state in model_list[start+1:end]:
                    param_data += model_state[param].data
                    num_ += 1.0
                param_data /= num_
                global_params[param] = param_data

        return global_params




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
                curr_model_dict = self.clients[clt].get_weights()
                curr_model = Net()
                curr_model.load_state_dict(curr_model_dict)
                curr_acc = eval(curr_model, self.test_loader, self.device)
                self.logs['val_acc_iter'].append(copy.deepcopy(curr_acc))

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
        self.index_level = [int( (self.B ** (i - 1) - 1) / (self.B - 1))   for i in range(1, self.h +1)]

    def train(self):

        for t in range(self.T):
            model_list = []

            shuffled_clts = super().shuffle_clients()
            print(shuffled_clts)
            for i, clt in enumerate(shuffled_clts):
                parent_index = int(np.floor(
                    (i - 1) / self.B))  # get parent index of clt, check my write up, parent of a node i, is [(i-1)/B]
                if parent_index >= 0:
                    parent_model_dict =  self.clients[shuffled_clts[parent_index]].get_weights()
                    self.clients[clt].model = Net()
                    self.clients[clt].model.load_state_dict(parent_model_dict)
                else:
                    if t >= 1:
                        self.clients[clt].model = copy.deepcopy(curr_model)

                self.clients[clt].train()


                model_list.append(dict(self.clients[clt].model.named_parameters()))


            for (start, end) in  zip(self.index_level [:-1], self.index_level[1:]):
                global_params = self.agg_model(model_list, start, end )
                curr_model = Net()
                curr_model.load_state_dict(global_params)
                curr_acc = eval(curr_model, self.test_loader, self.device)

                self.logs['val_acc_iter'].append(curr_acc)

            self.curr_model = copy.deepcopy(curr_model)
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

            global_params = self.agg_model(model_list, 0, len(model_list))
            curr_model = Net()
            curr_model.load_state_dict(global_params)
            self.curr_model = copy.deepcopy(curr_model)
            curr_acc = eval(self.curr_model, self.test_loader, self.device)
            self.logs['val_acc'].append(curr_acc)
            self.logs['val_acc_iter'].append(curr_acc)



class FedAvg(BaseFL):
    def __init__(self, configs=None):
        super().__init__(configs)
        self.R = self.num_clients // 2
        for clt_idx in range(self.num_clients):
            self.clients[clt_idx].fed_avg = True


    def train(self):
        for t in range(self.T):
            model_list = []
            shuffled_clts = super().shuffle_clients()
            for clt in shuffled_clts[:self.R]:
                if t >= 1:
                    self.clients[clt].model = copy.deepcopy(curr_model)
                self.clients[clt].train()
                model_list.append(dict(self.clients[clt].model.named_parameters()))

            global_params = self.agg_model(model_list, 0, len(model_list))

            curr_model = Net()
            curr_model.load_state_dict(global_params)
            self.curr_model = copy.deepcopy(curr_model)
            curr_acc = eval(self.curr_model, self.test_loader, self.device)
            print(curr_acc)
            self.logs['val_acc'].append(curr_acc)
            self.logs['val_acc_iter'].append(curr_acc)


class NewFedAvg(BaseFL):
    def __init__(self, configs=None):
        super().__init__(configs)
        self.R = self.num_clients // 5

        for clt_idx in range(self.num_clients):
            self.clients[clt_idx].fed_avg = False


    def train(self):
        for t in range(self.T):
            model_list = []
            shuffled_clts = super().shuffle_clients()
            for clt in shuffled_clts[:self.R]:
                if t >= 1:
                    self.clients[clt].model = copy.deepcopy(curr_model)
                self.clients[clt].train()
                model_list.append(dict(self.clients[clt].model.named_parameters()))

            global_params = self.agg_model(model_list, 0, len(model_list))

            curr_model = Net()
            curr_model.load_state_dict(global_params)
            self.curr_model = copy.deepcopy(curr_model)
            curr_acc = eval(self.curr_model, self.test_loader, self.device)
            print(curr_acc)
            self.logs['val_acc'].append(curr_acc)
            self.logs['val_acc_iter'].append(curr_acc)


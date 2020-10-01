# Federated training starting from here
from agent import *
import numpy as np


class BaseFL(object):
    def __init__(self, configs=None):
        default_configs = {
            'num_clients': 100,
            'T': 30,  # num outer epochs 30-40
            'B': 2,  # branch size of tree,
            'K':50,
            'params': {},
            'device':'cpu'
        }
        torch.manual_seed(0);
        self.curr_model = Net()

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

            #self.curr_model  = copy.deepcopy(curr_model)
            curr_acc = eval(curr_model, self.test_loader)
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
            agg_model = Net()
            for name, para in agg_model.named_parameters():
                para = torch.zeros(para.shape)

            shuffled_clts = super().shuffle_clients()
            print(shuffled_clts)
            for i, clt in enumerate(shuffled_clts):
                print('process {}'.format(i))
                parent_index = int(np.floor(
                    (i - 1) / self.B))  # get parent index of clt, check my write up, parent of a node i, is [(i-1)/B]
                if parent_index >= 0:
                    parent_model = copy.deepcopy( self.clients[shuffled_clts[parent_index]].model)
                    self.clients[clt].set_weights(parent_model)
                else:
                    if t >= 1:
                        self.clients[clt].model = copy.deepcopy(curr_model)

                self.clients[clt].train()

                if i >= self.index_leaf:
                    # aggregation step applied for leave nodes
                    with torch.no_grad():
                        for ((agg_name, agg_para), (clt_name, clt_para)) in zip(agg_model.named_parameters(),
                                                                                self.clients[
                                                                                    clt].model.named_parameters()):
                            if agg_name == clt_name:
                                agg_para += clt_para / self.num_leaves

            # self.set_weights(agg_model_dict)
            self.curr_model = copy.deepcopy(agg_model)
            curr_model = copy.deepcopy(agg_model)
            curr_acc = eval(self.curr_model, self.test_loader)
            self.logs['val_acc'].append(curr_acc)


class RingFL(BaseFL):
  def __init__(self, configs = None):
        super().__init__( configs)

  def train(self):
        for t in range(self.T):
            agg_model = Net()
            for name, para in agg_model.named_parameters():
                para = torch.zeros(para.shape)

            shuffled_clts = super().shuffle_clients()
            total_samples = float( np.sum([ self.clients[clt].num_train_samples for clt in shuffled_clts[:self.K]]))

            for clt in  shuffled_clts:
                if t >=1 :
                    self.clients[clt].set_weights(curr_model)
                self.clients[clt].train()
                with torch.no_grad():
                    for ((agg_name, agg_para), (clt_name, clt_para)) in zip(agg_model.named_parameters(), self.clients[clt].model.named_parameters()):
                        if agg_name == clt_name:
                            agg_para += clt_para * self.clients[clt].num_train_samples/total_samples

            self.curr_model = copy.deepcopy(agg_model)
            curr_model = copy.deepcopy(agg_model)
            curr_acc = eval(self.curr_model, self.test_loader, self.device)
            print(curr_acc)
            self.logs['val_acc'].append(curr_acc)

# class RingFL(FedAvg):
#     def __init__(self, configs=None):
#         super().__init__(configs)
#         self.K = self.num_clients
#
#     def train(self):
#         super().train()
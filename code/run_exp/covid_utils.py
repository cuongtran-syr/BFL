import sys
from keras.preprocessing.image import ImageDataGenerator

sys.path.insert(1, '/content/pytorch-dp/')
from torchdp import PrivacyEngine
from torchdp import PerSampleGradientClipper
import torch
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import csv
from pprint import pprint

import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import csv
from pprint import pprint


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def copy_model(model1, model2):
    """
    copy model1 to model2
    """
    params1 = model1.named_parameters()
    params2 = model2.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(param1.data)


def clear_backprops(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, "backprops_list"):
            del layer.backprops_list


def get_covid_model():
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    return model_conv


from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import copy
import numpy as np
from torchdp import PrivacyEngine


def get_augmented_data(train_loader, device):
    datagen = ImageDataGenerator(
        rotation_range=2,
        zoom_range=0.01,
        width_shift_range=0.01,
        height_shift_range=0.01)

    train_dataset = []

    for batch_idx, (data, target) in (enumerate(train_loader)):
        train_dataset.append([data, target])
        # print(type(target))
        for _ in range(9):
            if device == 'cuda':
                data_aug_x, data_aug_y = datagen.flow(data.cpu().numpy(), target.cpu().numpy()).next()
            else:
                data_aug_x, data_aug_y = datagen.flow(data, target).next()

            train_dataset.append([data_aug_x.reshape((1, 3, 224, 224)), target])

    random.shuffle(train_dataset)

    x_train = torch.cat([torch.FloatTensor(x[0]) for x in train_dataset])
    y_train = torch.cat([x[1] for x in train_dataset])

    return x_train, y_train


def eval(model, test_loader, device):
    """
    evaluation function -> similar to your test function 
    """
    model.eval()
    test_loss = 0
    correct = 0
    if device == 'cuda':
        model.to('cuda')
    with torch.no_grad():
        num_test_samples = 0
        for data, target in test_loader:
            if device == 'cuda':
                data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_test_samples += output.shape[0]

    test_loss /= num_test_samples

    return 100. * correct / num_test_samples


class Agent_CLF(object):
    def __init__(self, params):

        for key, val in params.items():
            setattr(self, key, val)
        self.logs = {'train_loss': [], 'eps': [], 'val_acc': []}
        torch.manual_seed(0)
        if not self.covid_model:
            self.model = Net()
        else:
            self.model = get_covid_model()
        if self.train_loader is None:
            if self.augmented == False:
                self.train_loader = DataLoader(dataset=data_utils.TensorDataset(self.x_train, self.y_train),
                                               batch_size=self.bs,
                                               shuffle=True)
            else:
                self.train_loader = DataLoader(dataset=data_utils.TensorDataset(self.x_train, self.y_train),
                                               batch_size=1,
                                               shuffle=True)

        if self.augmented == True:
            x_train, y_train = get_augmented_data(self.train_loader, self.device)
            self.train_loader = DataLoader(dataset=data_utils.TensorDataset(x_train, y_train), batch_size=self.bs,
                                           shuffle=True)

        self.num_train_samples = float(len(self.train_loader.dataset))
        self.num_run_epochs = 0
        self.random_idx = 0

    def set_weights(self, ref_model):
        if not self.covid_model:
            self.curr_model = Net()
        else:
            self.curr_model = get_covid_model()
        copy_model(ref_model, self.model)

    def get_weights(self):
        """
        get model  weights
        """
        w_dict = {}
        for name, param in self.model.named_parameters():
            w_dict[name] = copy.deepcopy(param)
        return w_dict

    def train(self):
        """
        train/update the curr model of the agent
        """
        optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        loss_func = nn.CrossEntropyLoss()

        if self.dp:
            self.model.zero_grad()
            optimizer.zero_grad()
            clear_backprops(self.model)

            privacy_engine = PrivacyEngine(
                self.model,
                batch_size=self.bs,
                sample_size=self.num_train_samples,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.sigma,
                max_grad_norm=self.C)
            privacy_engine.attach(optimizer)

        if self.device == 'cuda':
            self.model.to('cuda')
        self.model.train()
        for _ in range(self.epochs):
            num_batches = len(self.train_loader)
            start, end = 0, num_batches
            if self.fed_avg:
                start, end = self.random_idx, self.random_idx + 1
                self.random_idx += 1
                if self.random_idx >= num_batches:
                    self.random_idx = 0

            with torch.set_grad_enabled(True):
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    if start <= batch_idx < end:
                        if self.device == 'cuda':
                            data, target = data.to('cuda'), target.to('cuda')
                        optimizer.zero_grad()
                        output = self.model(data)
                        loss = loss_func(output, target)
                        loss.backward()
                        optimizer.step()
                        self.logs['train_loss'].append(copy.deepcopy(loss.item()))

            scheduler.step()
            self.lr = get_lr(optimizer)
            if self.fl_train is False:
                curr_acc = eval(self.model, self.test_loader, self.device)
                self.logs['val_acc'].append(copy.deepcopy(curr_acc))


import numpy as np

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import csv
from pprint import pprint


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class BaseFL(object):
    def __init__(self, configs=None):
        default_configs = {
            'num_clients': 100,
            'T': 20,  # num outer epochs 30-40
            'B': 2,  # branch size of tree,
            'params': {},
            'device': 'cpu'
        }

        if configs is not None:
            default_configs.update(configs)
            for key, val in default_configs.items():
                setattr(self, key, val)

        for key, val in default_configs.items():
            # set property for BaseFL object based on dictionary key-values.
            setattr(self, key, val)

        if not self.covid_model:
            self.curr_model = Net()
        else:
            self.curr_model = get_covid_model()

        self.clients = [Agent_CLF(self.params[i]) for i in range(self.num_clients)]
        self.logs = {'val_acc': [], 'val_acc_iter': []}

    def shuffle_clients(self):
        return np.random.permutation(self.num_clients)

    def set_weights(self, ref_model):
        """
        Set model
        """
        if not self.covid_model:
            self.curr_model = Net()
        else:
            self.curr_model = get_covid_model()
        copy_model(ref_model, self.curr_model)

    def get_weights(self):
        """
        get model  weights
        """
        w_dict = {}
        for name, param in self.curr_model.named_parameters():
            w_dict[name] = copy.deepcopy(param)
        return w_dict

    def agg_model(self, model_list, start, end):
        with torch.no_grad():
            global_params = {}
            for param in model_list[start]:
                param_data = model_list[start][param].data
                num_ = 1.0
                for model_state in model_list[start + 1:end]:
                    param_data += model_state[param].data
                    num_ += 1.0
                param_data /= num_
                global_params[param] = param_data

        return global_params


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

            if not self.covid_model:
                curr_model = Net()
            else:
                curr_model = get_covid_model()

            curr_model.load_state_dict(global_params, strict=False)
            self.curr_model = copy.deepcopy(curr_model)
            curr_acc = eval(self.curr_model, self.test_loader, self.device)
            print(curr_acc)
            self.logs['val_acc'].append(curr_acc)
            self.logs['val_acc_iter'].append(curr_acc)


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
                if not self.covid_model:
                    curr_model = Net()
                else:
                    curr_model = get_covid_model()

                curr_model.load_state_dict(curr_model_dict, strict=False)
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
        self.index_level = [int((self.B ** (i - 1) - 1) / (self.B - 1)) for i in range(1, self.h + 1)]

    def train(self):

        for t in range(self.T):
            model_list = []

            shuffled_clts = super().shuffle_clients()
            print(shuffled_clts)
            for i, clt in enumerate(shuffled_clts):
                parent_index = int(np.floor(
                    (i - 1) / self.B))  # get parent index of clt, check my write up, parent of a node i, is [(i-1)/B]
                if parent_index >= 0:
                    parent_model_dict = self.clients[shuffled_clts[parent_index]].get_weights()
                    if not self.covid_model:
                        self.clients[clt].model = Net()
                    else:
                        self.clients[clt].model = get_covid_model()

                    self.clients[clt].model.load_state_dict(parent_model_dict, strict=False)
                else:
                    if t >= 1:
                        self.clients[clt].model = copy.deepcopy(curr_model)

                self.clients[clt].train()

                model_list.append(dict(self.clients[clt].model.named_parameters()))

            for (start, end) in zip(self.index_level[:-1], self.index_level[1:]):
                global_params = self.agg_model(model_list, start, end)
                if not self.covid_model:
                    curr_model = Net()
                else:
                    curr_model = get_covid_model()
                curr_model.load_state_dict(global_params, strict=False)
                curr_acc = eval(curr_model, self.test_loader, self.device)

                self.logs['val_acc_iter'].append(curr_acc)

            self.curr_model = copy.deepcopy(curr_model)
            print(curr_acc)
            self.logs['val_acc'].append(curr_acc)


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

            with torch.no_grad():
                global_params = {}
                for param in model_list[0]:
                    param_data = model_list[0][param].data
                    for model_state in model_list[1:]:
                        param_data += model_state[param].data
                    param_data /= len(model_list)
                    global_params[param] = param_data

            if not self.covid_model:
                curr_model = Net()
            else:
                curr_model = get_covid_model()

            curr_model.load_state_dict(global_params, strict=False)
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

            with torch.no_grad():
                global_params = {}
                for param in model_list[0]:
                    param_data = model_list[0][param].data
                    for model_state in model_list[1:]:
                        param_data += model_state[param].data
                    param_data /= len(model_list)
                    global_params[param] = param_data

            if not self.covid_model:
                curr_model = Net()
            else:
                curr_model = get_covid_model()

            curr_model.load_state_dict(global_params, strict=False)
            self.curr_model = copy.deepcopy(curr_model)
            curr_acc = eval(self.curr_model, self.test_loader, self.device)
            print(curr_acc)
            self.logs['val_acc'].append(curr_acc)
            self.logs['val_acc_iter'].append(curr_acc)


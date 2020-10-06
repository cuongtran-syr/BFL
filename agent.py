from network import *
from utils import *
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import copy
import numpy as np
from torchdp import PrivacyEngine


class Agent_CLF(object):
    def __init__(self, params):

        for key, val in params.items():
            setattr(self, key, val)
        self.logs = {'train_loss': [], 'eps': [], 'val_acc': []}
        torch.manual_seed(0)
        self.model = Net()
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
        self.model = Net()
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
                start, end = self.random_idx,  self.random_idx +1
                self.random_idx += 1
                if self.random_idx >= num_batches:
                    self.random_idx = 0


            for batch_idx, (data, target) in enumerate(self.train_loader):
                if start <= batch_idx < end:
                    if self.device == 'cuda':
                        data, target = data.to('cuda'), target.to('cuda')
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()
                    self.logs['train_loss'].append(copy.deepcopy(loss.item()))

            scheduler.step()
            self.lr = get_lr(optimizer)
            if self.fl_train is False:
                curr_acc = eval(self.model, self.test_loader, self.device)
                self.logs['val_acc'].append(copy.deepcopy(curr_acc))

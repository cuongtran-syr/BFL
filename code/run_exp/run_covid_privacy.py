import sys
from covid_utils import *
import pickle
import argparse, time
file_path = '/home/cutran/Documents/federated_learning/res5/'
data_path = '/home/cutran/Documents/federated_learning/data/'

device = 'cpu'
if device == 'cuda':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)
from os import path


covid_data_path = '/home/cutran/Documents/federated_learning/data/COVID/data.pkl'
file_ = open(covid_data_path,'rb')
data_res = pickle.load(file_)

trainX = data_res['trainX']
trainY = data_res['trainY']
testX = data_res['testX']
testY =  data_res['testY']



def run_exp(model_choice, sigma, K, seed):
    file_name = file_path + 'augmented_{}_bfl_{}_{}_private_sigma_{}_K_{}_C20_.pkl'.format(model_choice, 'covid', sigma,
                                                                                           K, seed)
    if path.exists(file_name):
        print('EXIST {}'.format(file_name))
        return

    bs = 16
    test_loader = DataLoader(
        dataset=data_utils.TensorDataset(torch.FloatTensor(testX).permute(0, 3, 1, 2), torch.LongTensor(testY[:, 1])),
        batch_size=340,
        shuffle=True)

    temp_bs = 1440 // K
    train_loader = DataLoader(
        dataset=data_utils.TensorDataset(torch.FloatTensor(trainX).permute(0, 3, 1, 2), torch.LongTensor(trainY[:, 1])),
        batch_size=temp_bs,
        shuffle=True)

    default_params = {'lr': 1.0, 'augmented': True, 'bs': bs, 'gamma': 0.70, 'epochs': 1, 'fl_train': True,
                      'num_clients': K,
                      'dp': True, 'delta': 1e-5, 'sigma': sigma, 'C': 10, 'device': device, 'fed_avg': False,
                      'covid_model': True, 'device': device}

    params = {}
    for client_idx, (x_train, y_train) in enumerate(train_loader):
        params[client_idx] = copy.deepcopy(default_params)
        params[client_idx]['x_train'] = x_train
        params[client_idx]['y_train'] = y_train
        params[client_idx]['train_loader'] = None
        if model_choice == 'fedavg':
            params[client_idx]['fed_avg'] = True

    num_outer_epochs = 20
    num_iters = temp_bs // bs

    if model_choice == 'chain':
        fl_model = ChainFL(
            configs={'params': params, 'T': num_outer_epochs, 'B': 2, 'test_loader': test_loader, 'num_clients': K,
                     'covid_model': True, 'device': device})
    elif model_choice == 'tree':
        fl_model = TreeFL(
            configs={'params': params, 'T': num_outer_epochs, 'B': 2, 'test_loader': test_loader, 'num_clients': K,
                     'covid_model': True, 'device': device})
    elif model_choice == 'ring':
        fl_model = RingFL(
            configs={'params': params, 'T': num_outer_epochs, 'K': 100, 'test_loader': test_loader, 'num_clients': K,
                     'covid_model': True, 'device': device})
    elif model_choice == 'fedavg':
        num_rounds = int(num_outer_epochs * num_iters)
        fl_model = FedAvg(configs={'params': params, 'T': num_rounds, 'test_loader': test_loader, 'num_clients': K,
                                   'covid_model': True, 'device': device})
    else:
        fl_model = NewFedAvg(
            configs={'params': params, 'T': num_outer_epochs, 'test_loader': test_loader, 'num_clients': K,
                     'covid_model': True, 'device': device})

    fl_model.train()

    res = {}
    res['val_acc'] = copy.deepcopy(fl_model.logs['val_acc'])
    res['val_acc_iter'] = copy.deepcopy(fl_model.logs['val_acc_iter'])

    file_handle = open(file_name, 'wb')
    pickle.dump(res, file_handle)




def main():
   starttime = time.time()
   parser = argparse.ArgumentParser(description='Test')
   parser.add_argument('--model_choice', default='ring', type=str)
   parser.add_argument('--sigma', default= 0.5, type=float)
   parser.add_argument('--K', default=10, type=int)
   parser.add_argument('--seed', default=0, type=int)
   args = parser.parse_args()
   run_exp(args.model_choice,  args.sigma, args.K, args.seed)
   print('That took {} seconds'.format(time.time() - starttime))

if __name__ == "__main__":
    main()
from fltrain import *
import pickle
import argparse, time
file_path = '/home/cutran/Documents/federated_learning/res2/'
data_path = '/home/cutran/Documents/federated_learning/data/'

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


def run_exp(data, model_choice, sigma, K, seed):
    file_name = file_path + '{}_bfl_{}_{}_private_sigma_{}_K_{}_C20_.pkl'.format(model_choice, data, sigma, K, seed)

    if K  == 10:
        temp_bs = 6000
    else:
        temp_bs = 600
    if data == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=True, download=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size = temp_bs, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, download=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])), batch_size=10000, shuffle=True, **kwargs)
    else:

        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(data_path, train=True, download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ])), batch_size=temp_bs, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(data_path, train=False, download=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])), batch_size=10000, shuffle=True, **kwargs)

    default_params = {'lr': 1.0, 'bs': 64, 'gamma': 0.70, 'epochs': 1, 'fl_train':True,'num_clients':K,
                      'dp': True, 'delta': 1e-5, 'sigma': sigma, 'C': 20, 'device': 'cpu'}


    # if model_choice == 'central':
    #     default_params['train_loader'] = train_loader
    #     default_params['epochs'] = 40
    #     default_params['device'] = 'cpu'
    #     default_params['fl_train'] = False
    #     central_model = Agent_CLF(default_params)
    #     central_model.test_loader = test_loader
    #     central_model.train()
    #
    #     logs = central_model.logs['val_acc']
    #
    # else:
    params = {}
    for client_idx, (x_train, y_train) in enumerate(train_loader):
        params[client_idx] = copy.deepcopy(default_params)
        params[client_idx]['x_train'] = x_train
        params[client_idx]['y_train'] = y_train
        params[client_idx]['train_loader'] = None

    if model_choice == 'chain':
        fl_model = ChainFL(configs={'params': params, 'T': 15, 'B': 2, 'test_loader': test_loader, 'num_clients':K})
    elif model_choice == 'tree':
        fl_model = ChainFL(configs={'params': params, 'T': 15, 'B': 2, 'test_loader': test_loader,'num_clients':K})
    else:
        fl_model = RingFL(configs={'params': params, 'T': 15, 'K': 100, 'test_loader': test_loader,'num_clients':K})

    fl_model.train()

    logs = fl_model.logs['val_acc']

    file_handle = open(file_name, 'wb')
    pickle.dump(logs, file_handle)


def main():
   starttime = time.time()
   parser = argparse.ArgumentParser(description='Test')
   parser.add_argument('--model_choice', default='ring', type=str)
   parser.add_argument('--data', default='MNIST', type=str)
   parser.add_argument('--sigma', default= 0.5, type=float)
   parser.add_argument('--K', default=10, type=int)
   parser.add_argument('--seed', default=0, type=int)
   args = parser.parse_args()
   run_exp( args.data, args.model_choice,  args.sigma, args.K, args.seed)
   print('That took {} seconds'.format(time.time() - starttime))

if __name__ == "__main__":
    main()














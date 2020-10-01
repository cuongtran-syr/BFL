from fltrain import *
import pickle
import argparse, time
file_path = '/home/cutran/Documents/federated_learning/res/'
data_path = '/home/cutran/Documents/federated_learning/data/'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


def run_exp(data, model_choice, seed):

    file_name = file_path + '{}_bfl_{}_{}_private_sigma_0_5_C_20.pkl'.format(model_choice, data, seed)
    if model_choice == 'central':
        temp_bs = 64
    else:
        temp_bs = 600
    if data =='MNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=True, download= False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ])), batch_size= temp_bs, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, download=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])), batch_size= 10000, shuffle=True, **kwargs)
    else:

        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(data_path, train=True, download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ])), batch_size= temp_bs, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(data_path, train=False, download=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),batch_size= 10000,  shuffle=True, **kwargs)

    default_params = {'lr':1.0, 'bs':64, 'gamma':0.90, 'epochs':1,
                      'dp':True, 'delta':1e-5, 'sigma':0.5, 'C':20, 'device':'cpu'}

    logs = None

    if model_choice =='central':
        default_params['train_loader'] = train_loader
        default_params['epochs'] = 40
        default_params['fl_train'] = False
        central_model = Agent_CLF(default_params)
        central_model.test_loader = test_loader
        central_model.train()

        logs = central_model.logs['val_acc']

    else:
        default_params['fl_train'] = True
        params = {}
        for client_idx, (x_train, y_train) in enumerate(train_loader):
            params[client_idx] = copy.deepcopy(default_params)
            params[client_idx]['x_train'] = x_train
            params[client_idx]['y_train'] = y_train
            params[client_idx]['train_loader'] = None

        if model_choice =='chain':
            fl_model = ChainFL(configs={'params': params, 'T': 40, 'B': 2, 'test_loader': test_loader, 'device':'cpu'})
        elif model_choice =='tree':
            fl_model = ChainFL(configs={'params': params, 'T': 40, 'B': 2, 'test_loader': test_loader, 'device':'cpu'})
        else:
            fl_model = RingFL(configs={'params': params, 'T': 40, 'K': 50, 'test_loader': test_loader, 'device':'cpu'})

        fl_model.train()

        logs = fl_model.logs['val_acc']

    file_handle = open(file_name, 'wb')
    pickle.dump(logs, file_handle)


def main():
   starttime = time.time()
   parser = argparse.ArgumentParser(description='Test')
   parser.add_argument('--model_choice', default='central', type=str)
   parser.add_argument('--data', default='MNIST', type=str)
   parser.add_argument('--seed', default=0, type=int)
   args = parser.parse_args()
   run_exp( args.data, args.model_choice,  args.seed)
   print('That took {} seconds'.format(time.time() - starttime))

if __name__ == "__main__":
    #main_kfold()
    main()














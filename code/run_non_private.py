from fltrain import *
import pickle
import argparse, time
file_path = '/home/cutran/Documents/federated_learning/res3/'
data_path = '/home/cutran/Documents/federated_learning/data/'



def run_exp(data, model_choice, augment, bs,  K, seed):
    file_name = file_path + 'non_private_data_{}_{}_{}_{}_{}_{}.pkl'.format(model_choice, data, augment, bs, K, seed)
    if augment ==1:
        augmented = True
    else:
        augmented = False

    if K == 10:
        temp_bs = 6000
    elif K == 100:
        temp_bs = 600
    else:
        temp_bs = 1200
    if data == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=True, download=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size = temp_bs, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, download=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])), batch_size=10000, shuffle=True)
    else:

        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(data_path, train=True, download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ])), batch_size=temp_bs, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(data_path, train=False, download=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])), batch_size=10000, shuffle=True)

    default_params = {'lr': 1.0,'augmented':augmented, 'bs': bs, 'gamma': 0.70, 'epochs': 1, 'fl_train':True,'num_clients':K,
                      'dp': False, 'delta': 1e-5, 'sigma': 0.2, 'C': 20, 'device': 'cpu','fed_avg':False}


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
            configs={'params': params, 'T': num_outer_epochs, 'B': 2, 'test_loader': test_loader, 'num_clients': K})
    elif model_choice == 'tree':
        fl_model = TreeFL(
            configs={'params': params, 'T': num_outer_epochs, 'B': 2, 'test_loader': test_loader, 'num_clients': K})
    elif model_choice == 'ring':
        fl_model = RingFL(
            configs={'params': params, 'T': num_outer_epochs, 'K': 100, 'test_loader': test_loader, 'num_clients': K})
    else:
        num_rounds = int(num_outer_epochs * num_iters)
        fl_model = FedAvg(configs={'params': params, 'T': num_rounds, 'test_loader': test_loader, 'num_clients': K})

    fl_model.train()

    logs = fl_model.logs['val_acc']

    file_handle = open(file_name, 'wb')
    pickle.dump(logs, file_handle)


def main():
   starttime = time.time()
   parser = argparse.ArgumentParser(description='Test')
   parser.add_argument('--model_choice', default='ring', type=str)
   parser.add_argument('--data', default='MNIST', type=str)
   parser.add_argument('--augment', default= 0, type=int)
   parser.add_argument('--bs', default=16, type=int)
   parser.add_argument('--K', default=10, type=int)
   parser.add_argument('--seed', default=0, type=int)
   args = parser.parse_args()
   run_exp( args.data, args.model_choice,  args.augment, args.bs, args.K, args.seed)
   print('That took {} seconds'.format(time.time() - starttime))

if __name__ == "__main__":
    main()
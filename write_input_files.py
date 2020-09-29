dataset_list = [ 'MNIST','FMNIST']
model_choice_list = ['central', 'chain', 'tree', 'fed-avg']
file_ = open('run_privacy_exp.in', 'w')
for dataset in dataset_list:
    for model_choice in model_choice_list:
            for seed in range(10):
                                file_.write('{},{},{}\n'.format(dataset, model_choice, seed))

file_.close()

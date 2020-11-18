dataset_list = [ 'biased_MNIST','biased_FMNIST']
model_choice_list = ['fedavg']
K_list = [10, 50, 100]
sigma_list = [2.0]
file_ = open('./run_exp/run_misc_private_exp.in', 'w')
for dataset in dataset_list:
    for model_choice in model_choice_list:
        for sigma in sigma_list:
            for K in K_list:
                for seed in range(5):
                                file_.write('{},{},{},{},{}\n'.format(dataset, model_choice,sigma,K , seed))

file_.close()
# #
#
#
# dataset_list = [ 'MNIST','FMNIST']
# model_choice_list = [ 'tree','ring','chain','fedavg']
# K_list = [10, 50, 100]
# sigma_list = [0.5,  2.0, 3.0]
# file_ = open('./run_privacy_exp.in', 'w')
# for dataset in dataset_list:
#     for model_choice in model_choice_list:
#         for sigma in sigma_list:
#             for K in K_list:
#                 for seed in range(5):
#                                 file_.write('{},{},{},{},{}\n'.format(dataset, model_choice,sigma, K,seed))
#
# file_.close()



dataset_list = [ 'MNIST','FMNIST']
model_choice_list = [ 'fedavg']
K_list = [10, 50, 100]
sigma_list = [ 2.0]
file_ = open('./run_exp/run_privacy_exp.in', 'w')
for dataset in dataset_list:
    for model_choice in model_choice_list:
        for sigma in sigma_list:
            for K in K_list:
                for seed in range(5):
                                file_.write('{},{},{},{},{}\n'.format(dataset, model_choice,sigma, K,seed))

file_.close()


# dataset_list = [ 'MNIST','FMNIST']
# model_choice_list = [ 'tree','ring','chain','fedavg']
# K_list = [10, 50, 100]
# augment_list = [0, 1]
# bs_list = [16, 32, 64]
# file_ = open('run_non_private.in', 'w')
# for dataset in dataset_list:
#     for model_choice in model_choice_list:
#         for augment in augment_list:
#             for bs in bs_list:
#                 for K in K_list:
#                     for seed in range(5):
#                                     file_.write('{},{},{},{},{},{}\n'.format(dataset, model_choice, augment, bs, K,seed))
#
# file_.close()
#


# model_choice_list = [ 'fedavg']
# K_list = [5, 10, 20]
# sigma_list = [ 2.0]
# file_ = open('./run_exp/run_covid_privacy.in', 'w')
# for model_choice in model_choice_list:
#     for sigma in sigma_list:
#             for K in K_list:
#                 for seed in range(10):
#                                 file_.write('{},{},{},{}\n'.format(model_choice,sigma, K,seed))
#
# file_.close()
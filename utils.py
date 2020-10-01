import torch.nn as nn
import torch
from network import *

def eval(model, test_loader, device):
    """
    evaluation function -> similar to your test function 
    """
    model.eval()
    test_loss = 0
    correct = 0
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
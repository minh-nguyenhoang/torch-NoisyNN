import torch 
import torch.nn as nn 

from typing import Iterable
import math

##################################################################################################################################
###########      Taken from https://github.com/kytimmylai/NoisyNN-PyTorch/blob/main/noisy_resnet.py#L11      #####################
##################################################################################################################################

def quality_matrix(k, alpha=0.3):
    """r
    Quality matrix Q. Described in the eq (17) so that eps = QX, where X is the input. 
    Alpha is 0.3, as mentioned in Appendix D.
    """
    identity = torch.diag(torch.ones(k))
    shift_identity = torch.zeros(k, k) 
    for i in range(k):
        shift_identity[(i+1)%k, i] = 1
    opt = -alpha * identity + alpha * shift_identity
    return opt

def optimal_quality_matrix(k):
    """r
    Optimal Quality matrix Q. Described in the eq (19) so that eps = QX, where X is the input. 
    Suppose 1_(kxk) is torch.ones
    """
    return torch.diag(torch.ones(k)) * -k/(k+1) + torch.ones(k, k) / (k+1)

##################################################################################################################################

def __default_add_noise_fn__(layer_name: str, chosen_layers_name: Iterable[str], model: nn.Module, input, output: torch.Tensor):
    '''
    def add_noise_fn(layer_name: str, chosen_layers_name: Iterable[str], model: nn.Module, input, output):\n
        ***implement your logic here***\n 

    Args:
    - layer_name: Name of the current layer that is hooked with the NoisyNN instance.
    - chosen_layers_name: list of chosen layer activated in a forward batch.
    - model: the current layer specified by layer name (not the entire model).
    - input: the input of the layer from previous layer.
    - output: the output of the current layer, calculate as model.forward(input).
    '''

    if model.training:
        shape = output.shape
        old_shape = None
        if len(shape) == 3:
            old_shape = shape
            output = output.permute(0,2,1).view(shape[0],shape[2], math.sqrt(shape[1]), math.sqrt(shape[1]))
            shape = output.shape
        k = shape[-1]
        linear_noise = optimal_quality_matrix(k).to(output.device)
        output = linear_noise@output + output
        output = output.view(*shape)
        if old_shape is not None:
            output = output.view(shape[0], shape[1], -1).permute(0,2,1)

    return output

def __default_debug_fn__(layer_name: str, chosen_layers_name: Iterable[str], model: nn.Module, input, orignal_output, output):
    '''
    def debug_fn(layer_name: str, chosen_layers_name: Iterable[str], model: nn.Module, input, original_output, output):\n
        ***implement your logic here***\n 

    Args:
    - layer_name: Name of the current layer that is hooked with the NoisyNN instance.
    - chosen_layers_name: list of chosen layer activated in a forward batch.
    - model: the current layer specified by layer name (not the entire model).
    - input: the input of the layer from previous layer.
    - original_output: the output of the current layer, calculate as model.forward(input).
    - output: the output after going through noise transformation.
    '''
    if model.training:
        print(f'Layer {layer_name} is chosen!')
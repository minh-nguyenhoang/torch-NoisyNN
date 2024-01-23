import torch
import torch.nn as nn
from typing import Iterable, Optional, Callable
import random
import math
import warnings
from functools import wraps
from enum import Enum
from .default_function import __default_add_noise_fn__, __default_debug_fn__, safety_checker



'''
Inspired from https://github.com/kytimmylai/NoisyNN-PyTorch/tree/main, but I wrap it as a function to inject noise into any layer's output given its name.
There should be two way to implement the random layer choosing:
    - At each step, randomly choose a layer, register forward hook for that layer and remove it afther the forward pass. (Might cause performance issue? Need to revised.)
    - Register hook for each layer, then performing a random choosing operator at each forward pass. (This implementation should considerably harder but might overcone the overhead?)

Current implementation follow the second line, but I still wrap the first line implementation in a context manager noisy_nn(*args, **kwargs).
'''





##################################################################################################################################
####################################          CODE START HERE          ###########################################################
##################################################################################################################################


DEBUG_MODE = False
DEBUG_FALLBACK = False
SAFETY_CHECK = 1


class SAFETY_LEVEL(Enum):
    NONE = 0
    WARNING = 1
    CRITICAL = 2

class Chosen:
    '''
    just hold the value of the chosen layer name
    '''
    def  __init__(self, n_layers: int = 1) -> None:
        self.name = None
        self.n_layers = n_layers

def add_noise(name, chosen: Chosen, add_noise_fn: Optional[Callable]= None, debug_fn: Optional[Callable]= None):
    if debug_fn is None:
        debug_fn = __default_debug_fn__
    if add_noise_fn is None:
        add_noise_fn = __default_add_noise_fn__
    def hook(model: nn.Module, input, output: torch.Tensor):
        if name in chosen.name:
            if DEBUG_MODE or SAFETY_CHECK:
                original_output = output.clone()
            output = add_noise_fn(name, chosen.name, model, input, output)

            if DEBUG_MODE:
                try:
                    debug_fn(name, chosen.name, model, input, original_output, output)
                except:
                    warnings.warn(
                        f'''Debug function is not implemented correctly, please parse in correct function format as shown below:\n 
                                    def debug_fn(layer_name: str, chosen_layers_name: Iterable[str], model: nn.Module, input, original_output, output):\n
                                        ***implement your logic here***\n 

                                    Args:
                                    - layer_name: Name of the current layer that is hooked with the NoisyNN instance.
                                    - chosen_layers_name: list of chosen layer activated in a forward batch.
                                    - model: the current layer specified by layer name (not the entire model).
                                    - input: the input of the layer from previous layer.
                                    - original_output:the output of the current layer, calculate as model.forward(input).
                                    - output: the output after going through noise transformation.
                                ''')
                    
                    if DEBUG_FALLBACK:
                        warnings.warn("Due to exception occur with current debug function, fall back to default debug function.")
                        __default_debug_fn__(name, chosen.name, model, input, output)

            if SAFETY_CHECK != SAFETY_LEVEL.NONE:
                if SAFETY_CHECK == SAFETY_LEVEL.WARNING:
                    safety_checker(original_output, output, False)
                elif SAFETY_CHECK == SAFETY_LEVEL.CRITICAL:
                    safety_checker(original_output, output, True)
        return output
    return hook



def random_chooser(handles: dict, chosen: Chosen):
    def hook(model, input):
        chosen.name = []
        keys = list(handles.keys())
        keys.remove('choser')
        for _ in range(min(len(keys), chosen.n_layers)):           
            chosen.name.append(random.choice(keys))
            keys.remove(chosen.name[-1])
    return hook


def inject_noisy_nn(model: nn.Module, 
                    layers_name: Iterable[str], 
                    n_layers_inject_per_batch: int = 1,
                    add_noise_fn: Optional[Callable]= None,
                    debug_fn: Optional[Callable]= None,
                    inplace = True, 
                    verbose = True):
    if getattr(model, "is_noisy", False):
        if verbose:
            print(f"<--The model is already populated with NoisyNN instances! Please consider removing them before adding inject new NoisyNN instance.-->")
        return model
    
    if verbose:
        print('<--Trying to inject NoisyNN instances!-->')

    handles = {}
    chosen = Chosen(n_layers_inject_per_batch)
    layers_name = list(layers_name).copy()

    if not inplace:
        from copy import deepcopy
        model = deepcopy(model)
    max_idx = None
    for idx, (name, child) in enumerate(model.named_modules()):
        if name in layers_name:
            handles[name] = (child.register_forward_hook(add_noise(name= name, chosen= chosen, add_noise_fn= add_noise_fn, debug_fn= debug_fn))) 
            layers_name.remove(name)
            if verbose:
                print(f"---NoisyNN instance injected onto layer {name}.---")

    if verbose:
        if len(layers_name) > 0:
            repr = f'---Incompatible layer(s): {layers_name}'
        else:
            repr = "<--All chosen layers are injected with NoisyNN instance!-->"

        print(repr)

    handles["choser"] = model.register_forward_pre_hook(random_chooser(handles, chosen))
    setattr(model, "noisy_handles", handles)
    setattr(model, "is_noisy", True)
    return model    

def remove_noisy_nn(model: nn.Module, inplace = True, verbose = True):
    if not getattr(model, "is_noisy", False):
        if verbose:
            print("<--No NoisyNN instance left in the model!-->")
        return model
    
    print("<---Trying to remove NoisyNN!!!--->")

    if not inplace:
        from copy import deepcopy
        model = deepcopy(model)

    for name, handle in model.noisy_handles.items():
        handle.remove()
        if verbose and name != 'choser':
            print(f"---NoisyNN instance removed from layer {name}.---")

    delattr(model, 'is_noisy')
    delattr(model, 'noisy_handles')

    print("<--All injected NoisyNN instances have been removed!-->")

    return model

def disable_debug_mode():
    global DEBUG_MODE
    DEBUG_MODE = False

def enable_debug_mode():
    global DEBUG_MODE
    DEBUG_MODE = True

class debug_mode:
    '''
    Simple context manager and decorator for debugging NoisyNN. Usage:\n
    Case 1:
        ***Your code here***\n
        with debug_mode(...):
            ***Your code here***
        ***Your code here***\n

    Case 2:
        @debug_mode(...)\n
        def my_func(...):
            ***Your code here***
    '''
    def __init__(self, enabled = True, fallback = True) -> None:
        self.enabled = enabled
        self.fallback = fallback

    def __enter__(self):
        global DEBUG_MODE
        global DEBUG_FALLBACK
        self.original_mode = DEBUG_MODE
        if self.enabled:
            enable_debug_mode()
        else:
            disable_debug_mode()

        self.original_fallback = DEBUG_FALLBACK
        if self.fallback:
            DEBUG_FALLBACK = True
        else:
            DEBUG_FALLBACK = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        global DEBUG_MODE
        global DEBUG_FALLBACK
        DEBUG_MODE = self.original_mode
        DEBUG_FALLBACK = self.original_fallback

    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwds):
            with self:
                return func(*args, **kwds)
        return inner

class noisy_nn:
    '''
    Simple context manager and decorator for managing NoisyNN instances to a model in the scope. Usage:\n
    Case 1:
        ***Your code with *original model* here***\n
        with noisy_nn(model, ...):
            ***Your code with *modified model* here***
        ***Your code with *original model* here***\n

    Case 2:
        @noisy_nn(model, ...)\n
        def my_func(...):
            ***Your code with *modified model* here***
    '''
    def __init__(self, 
                    model: nn.Module, 
                    layers_name: Iterable[str], 
                    n_layers_inject_per_batch: int = 1,
                    add_noise_fn: Optional[Callable]= None,
                    debug_fn: Optional[Callable]= None,
                    verbose = True,
                    *args, **kwargs) -> None:
        self.model = model
        self.layers_name = layers_name
        self.n_layers_inject_per_batch = n_layers_inject_per_batch
        self.verbose = verbose
        self.add_noise_fn = add_noise_fn
        self.debug_fn = debug_fn

    def __enter__(self):
        self.model = inject_noisy_nn(self.model, self.layers_name, 
                                     self.n_layers_inject_per_batch, self.add_noise_fn, self.debug_fn,
                                     True, self.verbose)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return remove_noisy_nn(self.model, True, self.verbose)

    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwds):
            with self:
                return func(*args, **kwds)
        return inner
    
def disable_safety_check():
    global SAFETY_CHECK
    SAFETY_CHECK = 0

def enable_safety_check(strict_level:int = SAFETY_LEVEL.WARNING):
    global SAFETY_CHECK
    SAFETY_CHECK = strict_level

class safety_check:
    '''
    Simple context manager and decorator for controling safety checker of NoisyNN. Usage:\n
    Case 1:
        ***Your code here***\n
        with safety_check(level = ...):
            ***Your code here***
        ***Your code here***\n

    Case 2:
        @safety_check(level = ...)\n
        def my_func(...):
            ***Your code here***
    '''
    def __init__(self, level: int = SAFETY_LEVEL.WARNING) -> None:
        self.level = level

    def __enter__(self):
        global SAFETY_CHECK
        self.original_mode = SAFETY_CHECK
        if self.level:
            enable_safety_check(self.level)
        else:
            disable_safety_check()


    def __exit__(self, exc_type, exc_val, exc_tb):
        global SAFETY_CHECK
        SAFETY_CHECK = self.original_mode

    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwds):
            with self:
                return func(*args, **kwds)
        return inner

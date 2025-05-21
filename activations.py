import torch.nn as bkd

def linear(x): return x

import torch.nn as nn

def get(identifier):
    if identifier is None:
        return nn.Identity()
    if isinstance(identifier, str):
        return {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU(),
            "elu": nn.ELU(),
            "linear": nn.Identity(),
        }[identifier]
    if isinstance(identifier, nn.Module):
        return identifier
    raise TypeError("Invalid activation type")

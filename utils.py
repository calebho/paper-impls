"""Utility classes and functions"""
import time
import torch


class MyModuleList(torch.nn.ModuleList):
    """Wrapper for easily composing modules and activation functions."""

    def __init__(self, lst):
        self.layers = []
        self.activations = []
        for layer_activations in lst:
            self.layers.append(layer_activations[0])
            self.activations.append(layer_activations[1:])
        super().__init__(self.layers)

    def compose_modules(self):
        def f(x):
            for layer, activations in zip(self.layers, self.activations):
                x = layer(x)
                for activation in activations:
                    x = activation(x)
            return x
        return f


def compose(f, g):
    def h(*args, **kwargs):
        return g(f(*args, **kwargs))

    return h


def interleave(fst, snd):
    to_ret = []
    for x, y in zip(fst, snd):
        to_ret.append(x)
        to_ret.append(y)
    return to_ret


def timeit(f):
    def timed(*args, **kwargs):
        t0 = time.time()
        to_ret = f(*args, **kwargs)
        print(f'time elapsed {time.time() - t0:.4} s')
        return to_ret
    return timed

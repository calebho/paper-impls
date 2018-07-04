"""Utility functions"""


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

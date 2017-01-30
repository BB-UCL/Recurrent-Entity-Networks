"""
Generally useful helper functions
"""
import theano.tensor as T
import theano


def slice(item, start, slice_size):
    return item[start*slice_size:start*slice_size + slice_size]

def normalize(avector):
    return avector/(avector.norm(2))

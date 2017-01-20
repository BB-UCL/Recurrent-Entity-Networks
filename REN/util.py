"""
Generally useful helper functions
"""
import theano.tensor as T


def slice(item, start, slice_size):
    return item[start*slice_size:start*slice_size + slice_size]

def normalize(vector):
    return vector/(vector.norm(2))

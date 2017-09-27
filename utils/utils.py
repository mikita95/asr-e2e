import time
from functools import wraps
import numpy as np


def describe(func):
    """
    Wrapper for any function to log theirs estimating time
    Args:
        func: Some function

    Returns: The result value of the function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(func.__name__ + '...')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(str(func.__name__ + ' in ' + str(end - start) + ' s'))
        return result

    return wrapper


@describe
def count_params(model, mode='trainable'):
    if mode == 'all':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_op])
    elif mode == 'trainable':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_trainable_op])
    else:
        raise TypeError('mode should be all or trainable.')
    print('number of ' + mode + ' parameters: ' + str(num))
    return num


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representation of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
        dtype: type
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

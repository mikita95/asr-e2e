import os
import time
from functools import wraps

import numpy as np
import tensorflow as tf


# TODO: refactoring


def describe(func):
    # TODO: Need? Docs
    """

    Args:
        func:

    Returns:

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
def logging(model, logfile, errorRate, epoch=0, delta_time=0, mode='train'):
    if mode != 'train' and mode != 'test' and mode != 'config':
        raise TypeError('mode should be train or test or config.')
    logfile = logfile
    if mode == 'config':
        with open(logfile, "a") as myfile:
            myfile.write(str(model.config) + '\n')
    elif mode == 'train':
        with open(logfile, "a") as myfile:
            myfile.write(str(time.strftime('%X %x %Z')) + '\n')
            myfile.write("Epoch:" + str(epoch + 1) + ' ' + "train error rate:" + str(errorRate) + '\n')
            myfile.write("Epoch:" + str(epoch + 1) + ' ' + "train time:" + str(delta_time) + ' s\n')
    elif mode == 'test':
        logfile = logfile + '_TEST'
        with open(logfile, "a") as myfile:
            myfile.write(str(model.config) + '\n')
            myfile.write(str(time.strftime('%X %x %Z')) + '\n')
            myfile.write("test error rate:" + str(errorRate) + '\n')


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
    """Create a sparse representation of x.
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

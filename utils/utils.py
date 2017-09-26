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


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    """Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.
        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    """
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths

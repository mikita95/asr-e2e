import os
import time
from functools import wraps

import numpy as np
import tensorflow as tf

ALPHA_SIZE = 33

# TODO: refactoring


def describe(func):
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


def get_attrs(object, name):
    """

    Args:
        object:
        name:

    Returns:

    """
    assert type(name) == list, 'name must be a list'
    value = []
    for n in name:
        value.append(getattr(object, n, 'None'))
    return value


def set_attrs(object, attrsName, attrsValue):
    """

    Args:
        object:
        attrsName:
        attrsValue:

    Returns:

    """
    assert type(attrsName) == list, 'attrsName must be a list'
    assert type(attrsValue) == list, 'attrsValue must be a list'
    for name, value in zip(attrsName, attrsValue):
        object.__dict__[name] = value


def output_to_sequence(lmt):
    """

    Args:
        lmt:

    Returns:

    """
    sequences = []
    start = 0
    sequences.append([])
    for i in range(len(lmt[0])):
        if lmt[0][i][0] == start:
            sequences[start].append(lmt[1][i])
        else:
            start = start + 1
            sequences.append([])

    # here, we only print the first sequence of batch
    indexes = sequences[0]  # here, we only print the first sequence of batch
    seq = []
    for ind in indexes:
        if ind == 0:
            seq.append(' ')
        elif ind == ALPHA_SIZE + 1:
            seq.append("'")
        elif ind == ALPHA_SIZE + 2:
            pass
        else:
            seq.append(chr(ind + 96))
        seq = ''.join(seq)
        return seq


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


def list_to_sparse_tensor(target_list, mode):
    """

    Args:
        target_list:
        mode:

    Returns:

    """
    indices = []
    values = []
    assert mode == 'train' or mode == 'test', 'mode must be train or test'

    for tI, target in enumerate(target_list):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            values.append(val)
    shape = [len(target_list), np.asarray(indices).max(axis=0)[1] + 1]
    return np.array(indices), np.array(values), np.array(shape)


def get_edit_distance(hyp_arr, truth_arr, normalize, mode):
    """

    Args:
        hyp_arr:
        truth_arr:
        normalize:
        mode:

    Returns: Edit distance between sequences

    """
    graph = tf.Graph()
    with graph.as_default():
        truth = tf.sparse_placeholder(tf.int32)
        hyp = tf.sparse_placeholder(tf.int32)
        edit_dist = tf.reduce_sum(tf.edit_distance(hyp, truth, normalize=normalize))

    with tf.Session(graph=graph) as session:
        truth_test = list_to_sparse_tensor(truth_arr, mode)
        hyp_test = list_to_sparse_tensor(hyp_arr, mode)
        feed_dict = {truth: truth_test, hyp: hyp_test}
        dist = session.run(edit_dist, feed_dict=feed_dict)
    return dist


# TODO: needs refactoring
def data_lists_to_batches(inputList, targetList, batchSize, mode):
    ''' padding the input list to a same dimension, integrate all data into batchInputs
    '''
    assert len(inputList) == len(targetList)
    # dimensions of inputList:batch*39*time_length
    nFeatures = inputList[0].shape[0]
    maxLength = 0
    for inp in inputList:
        # find the max time_length
        maxLength = max(maxLength, inp.shape[1])
    # randIxs is the shuffled index from range(0,len(inputList))
    randIxs = np.random.permutation(len(inputList))
    start, end = (0, batchSize)
    dataBatches = []

    while end <= len(inputList):
        # batchSeqLengths store the time-length of each sample in a mini-batch
        batchSeqLengths = np.zeros(batchSize)
        # randIxs is the shuffled index of input list
        for batchI, origI in enumerate(randIxs[start:end]):
            batchSeqLengths[batchI] = inputList[origI].shape[-1]

        batchInputs = np.zeros((maxLength, batchSize, nFeatures))
        batchTargetList = []
        for batchI, origI in enumerate(randIxs[start:end]):
            # padSecs is the length of padding
            padSecs = maxLength - inputList[origI].shape[1]
            # numpy.pad pad the inputList[origI] with zeos at the tail
            batchInputs[:, batchI, :] = np.pad(inputList[origI].T, ((0, padSecs), (0, 0)), 'constant',
                                               constant_values=0)
            # target label
            batchTargetList.append(targetList[origI])
        dataBatches.append((batchInputs, list_to_sparse_tensor(batchTargetList, mode), batchSeqLengths))
        start += batchSize
        end += batchSize
    return (dataBatches, maxLength)


def load_batched_data(mfccPath, labelPath, batchSize, mode):
    '''returns 3-element tuple: batched data (list), maxTimeLength (int), and
       total number of samples (int)'''
    return data_lists_to_batches([np.load(os.path.join(mfccPath, fn)) for fn in os.listdir(mfccPath)],
                                 [np.load(os.path.join(labelPath, fn)) for fn in os.listdir(labelPath)],
                                 batchSize, mode) + \
           (len(os.listdir(mfccPath)),)


def _get_dims(shape):
    ''' get shape for initialization
    '''
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]
    return fan_in, fan_out


def dropout(x, keep_prob, is_training):
    """ Apply dropout to a tensor
    """
    return tf.contrib.layers.dropout(x, keep_prob=keep_prob, is_training=is_training)

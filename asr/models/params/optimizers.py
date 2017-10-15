import tensorflow as tf


def get(name):
    if name == 'adam':
        return tf.train.AdamOptimizer
    elif name == 'grad':
        return tf.train.GradientDescentOptimizer
    else:
        return None

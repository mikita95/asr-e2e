import nn.abstract_model as am
import tensorflow as tf


def _activation_summary(act):
    tf.summary.histogram(act.op.name + '/activations', act)
    tf.summary.scalar(act.op.name + '/sparsity', tf.nn.zero_fraction(act))


def _variable_on_cpu(name, shape, initializer, use_fp16):
    with tf.device('/cpu'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, wd_value, use_fp16):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                       mode='FAN_IN',
                                                       uniform=False,
                                                       seed=None,
                                                       dtype=dtype), use_fp16)
    if wd_value is not None:
        weight_decay = tf.cast(tf.scalar_mul(tf.nn.l2_loss(var), wd_value), tf.float32)  # CTC loss is in float32
        tf.add_to_collection('losses', weight_decay)
    return var


class LSTM(am.Model):

    def build_graph(self):
        num_hidden = 64
        num_layers = 3

        with tf.variable_scope('rnn') as scope:

            cells = []
            for i in range(num_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
                drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
                cells.append(drop_cell)
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

            rnn_outputs, _ = tf.nn.dynamic_rnn(cell=multi_cell,
                                               inputs=self.input,
                                               sequence_length=self.seq_lengths,
                                               scope='rnn',
                                               dtype=tf.float32)

            _activation_summary(rnn_outputs)

        # Linear layer(WX + b) is applied by CTC cost function.
        with tf.variable_scope('linear') as scope:
            W = tf.Variable(tf.truncated_normal([num_hidden,
                                                 self.num_classes],
                                                stddev=0.1))
            # Zero initialization
            # Tip: Is tf.zeros_initializer the same?
            b = tf.Variable(tf.constant(0., shape=[self.num_classes]))

            outputs = tf.reshape(rnn_outputs, [-1, num_hidden])

            # Doing the affine projection
            logits = tf.matmul(outputs, W) + b

            # Reshaping back to the original shape
            logits = tf.reshape(logits, [self.settings.batch_size, -1, self.num_classes])

            # Time major
            logits = tf.transpose(logits, (1, 0, 2))
            _activation_summary(logits)

        return logits

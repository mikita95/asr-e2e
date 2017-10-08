import nn.abstract_model as am
import tensorflow as tf

def _activation_summary(act):
    import re
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      act: Tensor
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', act.op.name)
    tf.summary.histogram(tensor_name + '/activations', act)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(act))


def _variable_on_cpu(name, shape, initializer, use_fp16):
    """Helper to create a Variable stored on cpu memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape,
                              initializer=initializer, dtype=dtype)
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
        weight_decay = tf.cast(tf.scalar_mul(tf.nn.l2_loss(var),
                                      wd_value),
                               tf.float32)  # CTC loss is in float32
        tf.add_to_collection('losses', weight_decay)
    return var

class LSTM(am.Model):
    def build_graph(self):
        num_hidden = 64
        num_layers = 3

        feat_len = self.input.get_shape().as_list()[-1]

        # convolutional layers
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay(
                'weights',
                shape=[11, feat_len, 1, 64],
                wd_value=None, use_fp16=False)

            feats = tf.expand_dims(self.input, dim=-1)
            conv = tf.nn.conv2d(feats, kernel,
                                [1, 2, 1, 1],
                                padding='SAME')
            # conv = tf.nn.atrous_conv2d(feats, kernel, rate=2, padding='SAME')
            biases = _variable_on_cpu('biases', [64],
                                      tf.constant_initializer(-0.05),
                                      use_fp16=False
                                      )
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv1)

            # dropout
            conv1_drop = tf.nn.dropout(conv1, 0.5)

        # recurrent layers
        with tf.variable_scope('rnn') as scope:

# Reshape conv output to fit rnn input                                  `                                                                                                                                                            GFXZ

            rnn_input = tf.reshape(conv1_drop, [self.settings.batch_size, -1,
                                                feat_len * 64])
            # Permute into time major order for rnn
            rnn_input = tf.transpose(rnn_input, perm=[1, 0, 2])
            # Make one instance of cell on a fixed device,
            # and use copies of the weights on other devices.
            cells = []
            for i in range(num_layers):
                cell = tf.nn.rnn_cell.LSTMCell(
                num_hidden)
                drop_cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=0.5)
                cells.append(drop_cell)
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

            seq_lens = self.seq_lengths
            rnn_outputs, _ = tf.nn.dynamic_rnn(multi_cell, rnn_input,
                                                   sequence_length=seq_lens, time_major=True,
                                                   scope='rnn', dtype=tf.float32)

            _activation_summary(rnn_outputs)

        # Linear layer(WX + b) - softmax is applied by CTC cost function.
        with tf.variable_scope('softmax_linear') as scope:
            weights = _variable_with_weight_decay(
                'weights', [num_hidden, self.num_classes],
                wd_value=None,
                use_fp16=False)
            biases = _variable_on_cpu('biases', [self.num_classes],
                                      tf.constant_initializer(0.0),
                                      False)
            logit_inputs = tf.reshape(rnn_outputs, [-1, cell.output_size])
            logits = tf.add(tf.matmul(logit_inputs, weights),
                            biases, name=scope.name)
            logits = tf.reshape(logits, [-1, self.settings.batch_size, self.num_classes])
            _activation_summary(logits)

        return logits
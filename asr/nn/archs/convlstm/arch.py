import tensorflow as tf

import asr.nn.model as am


class LSTM(am.Model):
    def build_graph(self):
        pass
        # TODO: complete
        # feat_len = feats.get_shape().as_list()[-1]
        #
        # # convolutional layers
        # with tf.variable_scope('conv1') as scope:
        #     kernel = am.Model._variable_with_weight_decay(
        #         'weights',
        #         shape=[11, feat_len, 1, params.num_filters],
        #         wd_value=None, use_fp16=False)
        #
        #     feats = tf.expand_dims(feats, dim=-1)
        #     conv = tf.nn.conv2d(feats, kernel,
        #                         [1, params.temporal_stride, 1, 1],
        #                         padding='SAME')
        #     # conv = tf.nn.atrous_conv2d(feats, kernel, rate=2, padding='SAME')
        #     biases = _variable_on_cpu('biases', [params.num_filters],
        #                               tf.constant_initializer(-0.05),
        #                               params.use_fp16)
        #     bias = tf.nn.bias_add(conv, biases)
        #     conv1 = tf.nn.relu(bias, name=scope.name)
        #     _activation_summary(conv1)
        #
        #     # dropout
        #     conv1_drop = tf.nn.dropout(conv1, params.keep_prob)
        #
        # # recurrent layers
        # with tf.variable_scope('rnn') as scope:
        #
        #     # Reshape conv output to fit rnn input
        #     rnn_input = tf.reshape(conv1_drop, [params.batch_size, -1,
        #                                         feat_len * params.num_filters])
        #     # Permute into time major order for rnn
        #     rnn_input = tf.transpose(rnn_input, perm=[1, 0, 2])
        #     # Make one instance of cell on a fixed device,
        #     # and use copies of the weights on other devices.
        #     cell = rnn_cell.CustomRNNCell(
        #         params.num_hidden, activation=tf.nn.relu6,
        #         use_fp16=params.use_fp16)
        #     drop_cell = tf.nn.rnn_cell.DropoutWrapper(
        #         cell, output_keep_prob=params.keep_prob)
        #     multi_cell = tf.nn.rnn_cell.MultiRNNCell(
        #         [drop_cell] * params.num_rnn_layers)
        #
        #     seq_lens = tf.div(seq_lens, params.temporal_stride)
        #     if params.rnn_type == 'uni-dir':
        #         rnn_outputs, _ = tf.nn.dynamic_rnn(multi_cell, rnn_input,
        #                                            sequence_length=seq_lens,
        #                                            dtype=dtype, time_major=True,
        #                                            scope='rnn',
        #                                            swap_memory=True)
        #     else:
        #         outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        #             multi_cell, multi_cell, rnn_input,
        #             sequence_length=seq_lens, dtype=dtype,
        #             time_major=True, scope='rnn',
        #             swap_memory=True)
        #         outputs_fw, outputs_bw = outputs
        #         rnn_outputs = outputs_fw + outputs_bw
        #     _activation_summary(rnn_outputs)
        #
        # # Linear layer(WX + b) - softmax is applied by CTC cost function.
        # with tf.variable_scope('softmax_linear') as scope:
        #     weights = _variable_with_weight_decay(
        #         'weights', [params.num_hidden, NUM_CLASSES],
        #         wd_value=None,
        #         use_fp16=params.use_fp16)
        #     biases = _variable_on_cpu('biases', [NUM_CLASSES],
        #                               tf.constant_initializer(0.0),
        #                               params.use_fp16)
        #     logit_inputs = tf.reshape(rnn_outputs, [-1, cell.output_size])
        #     logits = tf.add(tf.matmul(logit_inputs, weights),
        #                     biases, name=scope.name)
        #     logits = tf.reshape(logits, [-1, params.batch_size, NUM_CLASSES])
        #     _activation_summary(logits)
        #
        # return logits
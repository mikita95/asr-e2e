import tensorflow as tf

import asr.nn.model as am


class CONVLSTM(am.Model):
    def build_graph(self):
        pass
        # TODO: complete
        feat_len = self.input.get_shape().as_list()[-1]
        #
        # # convolutional layers
        with tf.variable_scope('conv1') as scope:
             kernel = am.Model._variable_with_weight_decay(
                 'weights',
                 shape=[18, feat_len, 1, int(self.configs['PARAMS']['num_filters'])],
                 wd_value=None, use_fp16=False)

             feats = tf.expand_dims(self.input, dim=-1)
             conv = tf.nn.conv2d(feats, kernel,
                                 [1, int(self.configs['PARAMS']['temporal_stride']), 1, 1],
                                 padding='SAME')
             conv = tf.nn.atrous_conv2d(feats, kernel, rate=2, padding='SAME')
             biases = am.Model._variable_on_cpu('biases', [int(self.configs['PARAMS']['num_filters'])],
                                       tf.constant_initializer(-0.05),
                                       use_fp16=False)
             bias = tf.nn.bias_add(conv, biases)
             conv1 = tf.nn.relu(bias, name=scope.name)
             am.Model._activation_summary(conv1)

             # dropout
             conv1_drop = tf.nn.dropout(conv1, float(self.configs['PARAMS']['keep_prob']))

        # # recurrent layers
        with tf.variable_scope('rnn') as scope:

             rnn_input = tf.reshape(conv1_drop, [self.batch_size, -1,
                                                 feat_len * int(self.configs['PARAMS']['num_filters'])])
        #     # Permute into time major order for rnn
             rnn_input = tf.transpose(rnn_input, perm=[1, 0, 2])
             cells = []
             num_layers = int(self.configs['PARAMS']['num_rnn_layers'])
             num_hidden = int(self.configs['PARAMS']['num_rnn_cells'])
             for i in range(num_layers):
                 cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
                 drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell, float(self.configs['PARAMS']['keep_prob']))
                 cells.append(drop_cell)
             multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
             outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    multi_cell, multi_cell, rnn_input,
                    sequence_length=self.seq_lengths,
                    time_major=True, scope='rnn', dtype=tf.float32, swap_memory=True)
             outputs_fw, outputs_bw = outputs
             rnn_outputs = outputs_fw + outputs_bw
             am.Model._activation_summary(rnn_outputs)

        with tf.variable_scope('softmax_linear') as scope:
            W = tf.Variable(tf.truncated_normal([num_hidden, self.num_classes],
                                                stddev=0.1))
            b = tf.Variable(tf.constant(0., shape=[self.num_classes]))

            outputs = tf.reshape(rnn_outputs, [-1, num_hidden])

            # Doing the affine projection
            logits = tf.matmul(outputs, W) + b

            # Reshaping back to the original shape
            logits = tf.reshape(logits, [self.batch_size, -1, self.num_classes])

            # Time major
            logits = tf.transpose(logits, (1, 0, 2))
            self._activation_summary(logits)

        return logits
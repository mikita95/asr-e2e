import tensorflow as tf

import asr.nn.model as am


class CONVLSTM(am.Model):
    def build_graph(self):
        feat_len = self.input.get_shape().as_list()[-1]
        #
        # # convolutional layers
        # with tf.variable_scope('conv1') as scope:
        #     feats = tf.expand_dims(self.input, dim=-1)
        #
        #     print(feats)
        #
        #     conv1 = tf.layers.conv2d(
        #         inputs=feats,
        #         filters=int(self.configs['PARAMS']['num_filters']),
        #         kernel_size=[2, 2],
        #         padding="same",
        #         activation=tf.nn.relu)
        #
        #     conv1_drop = tf.nn.dropout(conv1, float(self.configs['PARAMS']['keep_prob']))
        #
        # # # recurrent layers
        with tf.variable_scope('rnn') as scope:

             #rnn_input = tf.reshape(conv1_drop, [self.batch_size, -1,
             #                                    feat_len * int(self.configs['PARAMS']['num_filters'])])
        #     # Permute into time major order for rnn
             rnn_input = tf.transpose(self.input, perm=[1, 0, 2])
             cells = []
             num_layers = int(self.configs['PARAMS']['num_rnn_layers'])
             num_hidden = int(self.configs['PARAMS']['num_rnn_cells'])
             for i in range(num_layers):
                 cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
                 drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell, float(self.configs['PARAMS']['keep_prob']))
                 cells.append(drop_cell)
             multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
             rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    multi_cell, multi_cell, rnn_input,
                    sequence_length=self.seq_lengths,
                    time_major=True, scope='rnn', dtype=tf.float32, swap_memory=True)

        with tf.variable_scope('linear') as scope:
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
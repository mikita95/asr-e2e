import tensorflow as tf

import asr.nn.model as am


class LSTM(am.Model):
    def build_graph(self):
        with tf.variable_scope('rnn') as scope:
            cells = []
            num_layers = int(self.configs['PARAMS']['num_layers'])
            num_hidden = int(self.configs['PARAMS']['num_hidden'])
            for i in range(num_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
                drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell, float(self.configs['PARAMS']['output_keep_prob']))
                cells.append(drop_cell)

            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

            rnn_outputs, _ = tf.nn.dynamic_rnn(cell=multi_cell,
                                               inputs=self.input,
                                               sequence_length=self.seq_lengths,
                                               scope='rnn',
                                               dtype=tf.float32)

            self._activation_summary(rnn_outputs)

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

import nn.abstract_model as am
import tensorflow as tf


class LSTM(am.Model):
    def build_graph(self):
        num_hidden = 64
        num_layers = 3

        stacked_rnn = []
        for i in range(3):
            stacked_rnn.append(tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True))
        # Stacking rnn cells
        stack = tf.contrib.rnn.MultiRNNCell(stacked_rnn,
                                            state_is_tuple=True)



        # The second output is the last state and we will no use that

        outputs, _ = tf.nn.dynamic_rnn(stack, self.input, self.seq_lengths, dtype=tf.float32)

        shape = tf.shape(self.input)
        batch_s, max_time_steps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, num_hidden])

        # Truncated normal with mean 0 and stdev=0.1
        # Tip: Try another initialization
        # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
        W = tf.Variable(tf.truncated_normal([num_hidden,
                                             self.num_classes],
                                            stddev=0.1))
        # Zero initialization
        # Tip: Is tf.zeros_initializer the same?
        b = tf.Variable(tf.constant(0., shape=[self.num_classes]))

        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, self.num_classes])

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))

        return logits
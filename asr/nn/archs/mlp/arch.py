import asr.nn.model as am
import tensorflow as tf


class MLP(am.Model):
    def build_graph(self):
        import tensorflow.contrib.layers as layers

        num_hidden_layers = int(self.configs['PARAMS']['hidden_layers'])
        layers_dims = int(self.configs['PARAMS']['layers_dims'])
        hidden_layers = []

        with tf.name_scope('hidden'):
            hidden_layers.append(self.input)
            for i in range(num_hidden_layers):
                hidden_layers.append(
                    layers.fully_connected(inputs=hidden_layers[i],
                                           num_outputs=layers_dims,
                                           activation_fn=am.activation_function(self.configs['PARAMS']['activation'])))

        with tf.variable_scope('linear') as scope:
            W = tf.Variable(tf.truncated_normal([layers_dims, self.num_classes],
                                                stddev=0.1))
            b = tf.Variable(tf.constant(0., shape=[self.num_classes]))
            outputs = tf.reshape(hidden_layers[-1], [-1, layers_dims])

            # Doing the affine projection
            logits = tf.matmul(outputs, W) + b

            # Reshaping back to the original shape
            logits = tf.reshape(logits, [self.batch_size, -1, self.num_classes])

            # Time major
            logits = tf.transpose(logits, (1, 0, 2))
            self._activation_summary(logits)
        return logits

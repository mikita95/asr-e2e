from abc import ABCMeta, abstractmethod
from params.modes import Mode
import tensorflow as tf


class Model:
    __metaclass__ = ABCMeta

    def __init__(self, feature_input, seq_lengths, mode: Mode, num_classes, batch_size=None, config_file=None):
        self.input = feature_input
        self.seq_lengths = seq_lengths
        self.num_classes = num_classes

        self.batch_size = batch_size
        self.mode = mode

        self.configs = self._parse_config_file(config_file)

    @abstractmethod
    def build_graph(self):
        pass

    @staticmethod
    def _parse_config_file(config_file):
        if config_file is None:
            return None
        import configparser
        config = configparser.ConfigParser()
        config.read(config_file)
        return config

    @staticmethod
    def _activation_summary(act):
        tf.summary.histogram(act.op.name + '/activations', act)
        tf.summary.scalar(act.op.name + '/sparsity', tf.nn.zero_fraction(act))

    @staticmethod
    def _variable_on_cpu(name, shape, initializer, use_fp16=False):
        with tf.device('/cpu'):
            dtype = tf.float16 if use_fp16 else tf.float32
            var = tf.get_variable(name=name,
                                  shape=shape,
                                  initializer=initializer,
                                  dtype=dtype)
        return var

    @staticmethod
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
        var = Model._variable_on_cpu(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                       mode='FAN_IN',
                                                                       uniform=False,
                                                                       seed=None,
                                                                       dtype=dtype),
            use_fp16=use_fp16)

        if wd_value is not None:
            weight_decay = tf.cast(tf.scalar_mul(tf.nn.l2_loss(var), wd_value), tf.float32)
            tf.add_to_collection('losses', weight_decay)
        return var

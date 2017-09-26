import tensorflow as tf

from models.abstract_model import Model
from utils.utils import describe
from tensorflow.python.ops import ctc_ops as ctc

#TODO: implement methods

def build_residual_block(inpt, out_channels, down_sample=False, projection=False, name='block1'):
    pass


def build_resnet(inpt, maxTimeSteps, depth, width, num_class):
    pass


class ResNet(Model):

    @describe
    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputX = tf.placeholder(tf.float32,
                                         shape=(self.maxTimeSteps, args.batch_size, args.num_feature))  # [maxL,32,39]
            inputXrs = tf.reshape(self.inputX, [-1, args.num_feature])
            self.inputList = tf.split(inputXrs, self.maxTimeSteps, 0)  # convert inputXrs from [32*maxL,39] to [32,maxL,39]

            self.targetIxs = tf.placeholder(tf.int64)
            self.targetVals = tf.placeholder(tf.int32)
            self.targetShape = tf.placeholder(tf.int64)
            self.targetY = tf.SparseTensor(self.targetIxs, self.targetVals, self.targetShape)
            self.seqLengths = tf.placeholder(tf.int32, shape=(args.batch_size))
            depth = 10
            width = 8
            self.config = {'name': 'residual network',
                           'num_layer': depth,
                           'num_featuremap': width,
                           'num_class': args.num_class,
                           'optimizer': args.optimizer,
                           'learning rate': args.learning_rate
                           }

            inpt = tf.reshape(self.inputX, [args.batch_size, maxTimeSteps, args.num_feature, 1])
            conv_output = build_resnet(inpt, maxTimeSteps, depth, width, args.num_class)
            self.loss = tf.reduce_mean(ctc.ctc_loss(self.targetY, conv_output, self.seqLengths))
            self.optimizer = args.optimizer(args.learning_rate).minimize(self.loss)
            self.logitsMaxTest = tf.slice(tf.argmax(conv_output, 2), [0, 0], [self.seqLengths[0], 1])
            self.predictions = tf.to_int32(ctc.ctc_beam_search_decoder(conv_output, self.seqLengths)[0][0])
            self.errorRate = tf.reduce_sum(
                tf.edit_distance(self.predictions, self.targetY, normalize=False)) / tf.to_float(
                tf.size(self.targetY.values))
            self.initial_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2, keep_checkpoint_every_n_hours=1)
            #self.logfile = args.log_dir + str(
            #    datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S') + '.txt').replace(' ',
            #                                                                                               '').replace(
            #    '/', '')
            self.var_op = tf.global_variables()
            self.var_trainable_op = tf.trainable_variables()
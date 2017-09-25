import argparse
import utils.data_processor as dp

FLAGS = None


def run():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='trainer', description='Script to train and test models')

    parser.add_argument('--mode',
                        type=str,
                        help='Running mode',
                        choices=['train', 'test'],
                        default='train')

    parser.add_argument('--data_path',
                        type=str,
                        help='Path to data dir')

    parser.add_argument('--model',
                        help='Name of neural network model',
                        type=str)

    parser.add_argument('--how_many_training_steps',
                        type=str,
                        default='15000,3000',
                        help='How many training loops to run')

    parser.add_argument('--eval_step_interval',
                        type=int,
                        default=400,
                        help='How often to evaluate the training results.')

    parser.add_argument('--learning_rate',
                        type=str,
                        default='0.001,0.0001',
                        help='How large a learning rate to use when training.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='How many items to train with at once')

    parser.add_argument('--summaries_dir',
                        type=str,
                        default='/tmp/retrain_logs',
                        help='Where to save summary logs for TensorBoard.')

    parser.add_argument('--train_dir',
                        type=str,
                        default='/tmp/speech_commands_train',
                        help='Directory to write event logs and checkpoint.')

    parser.add_argument('--save_step_interval',
                        type=int,
                        default=100,
                        help='Save model checkpoint every save_steps.')

    parser.add_argument('--start_checkpoint',
                        type=str,
                        default='',
                        help='If specified, restore this pretrained model before any training.')
    run()

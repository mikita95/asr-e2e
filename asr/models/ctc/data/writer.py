import asr.utils.data.examples.writer as wr
import asr.utils.data.examples.features.selector as fsb
import tensorflow as tf
import argparse
import sys


def main(_):
    import asr.models.ctc.data.labels.handler as lh
    tf.logging.set_verbosity(tf.logging.INFO)

    writer = wr.Writer(features_selector=fsb.get_feature_selector(selector_name=FLAGS.mode),
                       labels_handler=lh.CTCLabelsHandler)

    writer.write(data_dir=FLAGS.data_dir,
                 record_file_path=FLAGS.save_tfrecord_path,
                 feature_settings=vars(FLAGS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='data_process', description='Script to process data')

    parser.add_argument("--data_dir",
                        help="Directory of dataset",
                        type=str)

    parser.add_argument("--format",
                        help="Format of files",
                        choices=['wav', 'mp3'],
                        type=str,
                        default='wav')

    parser.add_argument("--save_tfrecord_path",
                        help="Path to save tfrecord file",
                        type=str)

    parser.add_argument("--selectors",
                        help="Type of feature selector",
                        choices=list(map(int, fsb.Selector)),
                        type=str,
                        default='mfcc')

    feature_settings_group = parser.add_argument_group('feature_settings')

    feature_settings_group.add_argument("--rate",
                                        help="Sample rate of the audio files",
                                        type=int,
                                        default=16000)

    feature_settings_group.add_argument("--channels",
                                        help="Number of channels of the audio files",
                                        type=int,
                                        default=1)

    feature_settings_group.add_argument("--winlen", type=float, default=0.025)
    feature_settings_group.add_argument("--winstep", type=float, default=0.01)
    feature_settings_group.add_argument("--numcep", type=int, default=13)
    feature_settings_group.add_argument("--nfilt", type=int, default=26)
    feature_settings_group.add_argument("--nfft", type=int, default=512)
    feature_settings_group.add_argument("--lowfreq", type=int, default=0)
    feature_settings_group.add_argument("--highfreq", type=int, default=None)
    feature_settings_group.add_argument("--ceplifter", type=int, default=22)
    feature_settings_group.add_argument("--preemph", type=float, default=0.97)
    feature_settings_group.add_argument("--appendEnergy", type=bool, default=True)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
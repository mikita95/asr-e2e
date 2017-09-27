import numpy as np
import utils.data_processor as dp
import utils.utils as ut


SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('а') - 1  # 0 is reserved to space


def convert_inputs_to_ctc_format(features):
    pass


def convert_input_to_ctc_format(feature_audio, target_text):
    # TODO: Fill docs
    """
    Args:
        feature_audio:
        target_text:

    Returns:

    """
    # Transform in 3D array
    train_inputs = np.asarray(feature_audio[np.newaxis, :])
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    train_seq_len = [train_inputs.shape[1]]

    # Get only the words between [а-я] and replace period for none
    original = ' '.join(target_text.strip().lower().split(' ')).replace('.', '').replace('?', '').replace(',', ''). \
        replace("'", '').replace('!', '').replace('-', '')

    targets = original.replace(' ', '  ')
    targets = targets.split(' ')

    # Adding blank label
    targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

    # Transform char into index
    targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                             for x in targets])

    # Creating sparse representation to feed the placeholder
    train_targets = ut.sparse_tuple_from([targets])

    return train_inputs, train_targets, train_seq_len, original


def handle_batch(batch):
    feature_batch = map(lambda x: [dp.parse_feature_file(batch[0])], batch)
    feature_batch = dp.pad_features(feature_batch)

    return convert_inputs_to_ctc_format(feature_batch)
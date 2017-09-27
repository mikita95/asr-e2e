import numpy as np
import utils.data_processor as dp
import utils.utils as ut


SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('а') - 1  # 0 is reserved to space


def convert_input_to_ctc_format(feature_vector, target_text):
    """
    Args:
        feature_vector: numpy array of dims [num_cep x seq_length]
        target_text: text of the corresponding label

    Returns: tuple of (feature_tensor, target, seq_length, original)
             where feature_tensor is a tensor [1 x seq_length x num_cep],
                   target is an array [number_of_characters + blanks]
                   seq_length is a list object with one element equals length of feature_vector length
                   original is a list of original words parsed from target_text
    """
    # Transform into 3D array
    feature_tensor = np.asarray(feature_vector[np.newaxis, :])  # [1 x seq_length x num_cep]
    feature_tensor = (feature_tensor - np.mean(feature_tensor)) / np.std(feature_tensor)  # normalize
    seq_length = [feature_tensor.shape[1]]

    # Get only the words between [а-я] and replace period for none
    original = ' '.join(target_text.strip().lower().split(' ')).replace('.', '').replace('?', '').replace(',', ''). \
        replace("'", '').replace('!', '').replace('-', '')

    targets = original.replace(' ', '  ')
    targets = targets.split(' ')  # list of dims [words + blanks]

    # Adding blank label
    targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])  # array [number_of_characters + blanks]

    # Transform char into index, array [number_of_characters + blanks]
    target = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])

    return feature_tensor, target, seq_length, original


def union_feature_tensors(feature_tensors):
    return np.concatenate(tuple(feature_tensors), axis=0)


def convert_inputs_to_ctc_format(batch):
    feature_tensors = []
    seq_lengths = []
    originals = []
    targets = []

    for i in range(0, len(batch)):
        feature_tensor, target, seq_length, original = \
            convert_input_to_ctc_format(batch[i][0], batch[i][1])

        feature_tensors.append(feature_tensor)
        seq_lengths.extend(seq_length)
        originals.append(original)
        targets.append(target)

    feature_tensors = union_feature_tensors(feature_tensors)  # tensor [batch_size x max_seq_length x num_cep]
    # Creating sparse representation to feed the placeholder, tuple of (indices, values, shape)
    sparse_targets = ut.sparse_tuple_from(targets)
    return feature_tensors, sparse_targets, seq_lengths, originals


def handle_feature_vectors_batch(feature_vectors_batch):
    padded_batch = dp.pad_feature_vectors(feature_vectors_batch)  # list of [[max(seq_length) x num_cep] arrays, text]

    return convert_inputs_to_ctc_format(padded_batch)


def handle_batch(batch):
    """
    Convert batches to needed ctc format
    Args:
        batch: a list of pairs <path_to_feature_file, target_text>

    Returns: feature_tensors, sparse_targets, seq_lengths, originals
             where feature_tensors is a tensor [batch_size x max_seq_length x num_cep]
             sparse_targets is a concatenated sparse tensor of targets
             seq_lengths is a list of seq_lengths
             originals is a list of lists containing original words

    """
    feature_vectors_batch = \
        map(lambda x: [dp.parse_example_file(x[0]), x[1]], batch)  # list of [[seq_length x num_cep] arrays, text]
    return handle_feature_vectors_batch(feature_vectors_batch)

from abc import ABC, abstractmethod
from enum import Enum
import tensorflow as tf
import numpy as np


class Handler(Enum):
    INDEXER = 'indexer'


class LabelsHandler(ABC):

    @abstractmethod
    def encode(self, label: object) -> tf.train.Feature:
        pass

    @abstractmethod
    def decode(self, label):
        pass


class SentencesLabelsHandler(LabelsHandler):

    def __init__(self, alphabet_file):
        def _parse_alphabet_file(alphabet_file_path):
            if alphabet_file is None:
                return None
            import configparser
            import ast
            config = configparser.ConfigParser()
            config.read(alphabet_file_path, encoding='utf8')
            alphabet = dict(config.items('ALPHABET'))

            for key, value in config['SPECIAL_SYMBOLS'].items():
                if key in alphabet:
                    alphabet[ast.literal_eval(value)] = alphabet.pop(key)
            return alphabet

        self.alphabet = _parse_alphabet_file(alphabet_file)

    def get_alphabet_size(self):
        return len(self.alphabet)

    @abstractmethod
    def encode(self, label: object) -> tf.train.Feature:
        pass

    @abstractmethod
    def decode(self, label):
        pass


class IndexerSentencesLabelsHandler(SentencesLabelsHandler):

    def _convert_label_to_index_format(self, label_text: str) -> np.ndarray:
        def _normalize_label_text(text: str) -> str:
            return text.lower().replace('.', '').replace('?', '').replace(',', ''). \
                replace("'", '').replace('!', '').replace('-', '')

        original = _normalize_label_text(label_text)

        label = np.asarray([int(self.alphabet[c]) for c in original if c in self.alphabet])
        return label

    def encode(self, label: str) -> tf.train.Feature:
        converted_label = self._convert_label_to_index_format(label)
        label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=converted_label))
        return label_feature

    def decode(self, sequence):
        inv_alphabet = {v: k for k, v in self.alphabet.items()}
        return ''.join([inv_alphabet[str(c)] for c in sequence])


def get_labels_handler(alphabet_file, handler_name='indexer'):
    if handler_name == Handler.INDEXER.value:
        return IndexerSentencesLabelsHandler(alphabet_file=alphabet_file)
    else:
        raise TypeError('No such labels handler.')
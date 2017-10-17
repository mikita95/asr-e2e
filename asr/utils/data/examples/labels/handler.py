from abc import ABC, abstractmethod
from enum import Enum
from asr.utils.data.examples.labels.indexer import IndexerLabelsHandler


class Handler(Enum):
    INDEXER = 'indexer'


def get_labels_handler(alphabet_file, handler_name='indexer'):
    if handler_name == Handler.INDEXER:
        return IndexerLabelsHandler(alphabet_file=alphabet_file)
    else:
        raise TypeError('No such labels handler.')


class LabelsHandler(ABC):

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
    def encode(self, label):
        pass

    @abstractmethod
    def decode(self, sequence):
        pass

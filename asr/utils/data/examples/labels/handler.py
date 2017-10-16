from abc import ABCMeta, abstractmethod
import pkg_resources


class LabelsHandler:
    __metaclass__ = ABCMeta

    def __init__(self, alphabet_file):
        self.alphabet = self._parse_alphabet_file(alphabet_file)

    def _parse_alphabet_file(self, alphabet_file):
        if alphabet_file is None:
            return None
        import configparser
        import ast
        config = configparser.ConfigParser()
        config.read(alphabet_file, encoding='utf8')
        alphabet = dict(config.items('ALPHABET'))

        for key, value in config['SPECIAL_SYMBOLS'].items():
            if key in alphabet:
                alphabet[ast.literal_eval(value)] = alphabet.pop(key)
        return alphabet

    def get_alphabet_size(self):
        return len(self.alphabet)

    @abstractmethod
    def encode(self, label):
        pass

    @abstractmethod
    def decode(self, sequence):
        pass

from abc import ABCMeta, abstractmethod


class LabelsHandler:
    __metaclass__ = ABCMeta

    def __init__(self, alphabet_file):
        self.alphabet = self._parse_alphabet_file(alphabet_file)

    def _parse_alphabet_file(self, alphabet_file):
        if alphabet_file is None:
            return None
        import configparser
        config = configparser.ConfigParser()
        config.read(alphabet_file)
        alphabet = config['ALPHABET']
        for key, value in config['SPECIAL_SYMBOLS'].items():
            if key in alphabet:
                alphabet[value] = alphabet.pop(key)
        return alphabet

    def get_alphabet_size(self):
        return len(self.alphabet)

    @abstractmethod
    def handle(self, label):
        pass

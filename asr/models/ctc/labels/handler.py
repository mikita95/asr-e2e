import asr.utils.data.examples.labels.handler as alh


class CTCLabelsHandler(alh.LabelsHandler):

    def _convert_label_to_ctc_format(self, label_text):
        import numpy as np

        def _normalize_label_text(text):
            return text.lower().replace('.', '').replace('?', '').replace(',', ''). \
                replace("'", '').replace('!', '').replace('-', '')

        original = _normalize_label_text(label_text)

        label = np.asarray([self.alphabet[c] for c in original if c in self.alphabet])

        return label

    def encode(self, label):
        return self._convert_label_to_ctc_format(label)

    def decode(self, sequence):
        inv_alphabet = {v: k for k, v in self.alphabet.items()}
        return ''.join([inv_alphabet[str(c)] for c in sequence])
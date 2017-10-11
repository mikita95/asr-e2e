import src.main.utils.data.examples.labels.handler as alh


class CTCLabelsHandler(alh.LabelsHandler):

    def _convert_label_to_ctc_format(self, label_text):
        import numpy as np

        def _normalize_label_text(text):
            return ' '.join(text.strip().lower().split(' ')).replace('.', '').replace('?', '').replace(',', ''). \
                replace("'", '').replace('!', '').replace('-', '')

        original = _normalize_label_text(label_text)

        label = np.asarray([self.alphabet[c] for c in original if c in self.alphabet])

        return label

    def handle(self, label):
        return self._convert_label_to_ctc_format(label)

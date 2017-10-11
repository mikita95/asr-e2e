from distutils.core import setup

setup(
    name='asr-e2e',
    version='0.1',
    packages=['nn', 'nn.archs', 'nn.archs.mlp', 'nn.archs.lstm', 'nn.archs.lstm.configs', 'utils', 'utils.data',
              'utils.data.examples', 'utils.data.examples.labels', 'utils.data.examples.features', 'models',
              'models.ctc', 'models.ctc.labels', 'models.params'],
    url='https://github.com/mikita95/asr-e2e',
    license='',
    author='Nikita_Markovnikov',
    author_email='niklemark@gmail.com',
    description='Automatic speech recognition system using end-to-end approach for Russian speech',
    requires=['tensorflow', 'python_speech_features', 'numpy']
)

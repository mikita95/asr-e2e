from setuptools import setup, find_packages
setup(
    name='asr-e2e',
    version='',
    packages=find_packages(exclude=["tests*"]),
    package_data={'asr': ['resources/*'], 'tests': ['resources/*']},
    include_package_data=True,
    test_suite='tests',
    url='',
    license='',
    author='Nikita_Markovnikov',
    author_email='',
    description=''
)

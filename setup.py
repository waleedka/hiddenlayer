from setuptools import setup

setup(
    name = 'HiddenLayer',
    packages = ['hiddenlayer'],
    version = '0.1',
    license="MIT",
    description = 'A light-weight alternative to TensorBoard for smaller deep learning projects',
    author = 'Waleed Abdulla <waleed.abdulla@gmail.com>, Phil Ferriere <pferriere@hotmail.com>',
    url = 'https://github.com/waleedka/hiddenlayer',
    classifiers = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 4 - Beta',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Visualization',

    # Pick your license as you wish (should match "license" above)
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3.5',
    ],
)
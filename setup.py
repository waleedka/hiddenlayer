import setuptools

setuptools.setup(
    name = 'hiddenlayer',
    # packages = ['hiddenlayer'],
    packages = setuptools.find_packages(),
    version = '0.2',
    license="MIT",
    description = 'Neural network graphs and training metrics for PyTorch and TensorFlow',
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

    'Operating System :: OS Independent',
    ],
)

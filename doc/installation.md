# Installation

Philote-Python is a pure Python library. It ships with generated python for grpc which can be
generated during the development process. For a standard installation, however, it can just be installed via pip.


## Requirements

The development process requires the following tools to be installed:

- grpcio-tools
- protoletariat
- importlib.resources

Additionally, the following dependencies are required by Philote MDO and will be
installed automatically during the installation process:

- numpy
- grpcio

To run the unit and integration tests, you will need:

- openmdao (can be found [here](https://github.com/OpenMDAO/OpenMDAO) or installed via pip)


## Compiling Definitions and Installation

Older versions of this library featured a two-step build process. This has since
been simplified. To install the package run pip:

    pip install <path/to/Philote-Python>

or

    pip install -e <path/to/Philote-Python>

for an editable install. Note, that <path/to/Philote-Python> is the path to the
repository root directory (the one containing pyproject.toml). Often, people
install packages when located in that directory, making the corresponding
command:

    pip install .
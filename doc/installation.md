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

The easiest way for users to install this library is via pip using the PyPI package:

    pip install philote-mdo

If you need or want to install the package from the repository, you can doe this using pip:

    pip install <path/to/Philote-Python>

or

    pip install -e <path/to/Philote-Python>

for an editable install. Note, that <path/to/Philote-Python> is the path to the
repository root directory (the one containing pyproject.toml). Often, people
install packages when located in that directory, making the corresponding
command:

    pip install .

Unlike earlier version, the package is distributed with generated gRPC python files. This means that you do not need to
have grpciotools or protoc installed when using Philote-Python. If you are doing development work, specifically when you
are adding new gRPC features, you will need to regenerate the gRPC files. To do this, run the following command from the
repository root directory:

    python utils/compile_proto.py
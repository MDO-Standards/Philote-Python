# Installation

Philote-Python is a pure Python library. However, the installation process requires a few extra steps,
as the gRPC/protobuf definitions must first be compiled into 


## Requirements

The installation process requires the following tools to be installed:

- grpcio-tools
- protoletariat
- importlib.resources

Additionally, the following dependencies are required by Philote MDO and will be
installed automatically during the installation process:

- numpy
- grpcio

To run the unit and integration tests, you will need:

- openmdao (can be found [here](https://github.com/OpenMDAO/OpenMDAO) or installed via pip)
- csdl (can be found [here](https://github.com/LSDOlab/csdl))
- csdl python backend (can be found [here](https://github.com/LSDOlab/python_csdl_backend))


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
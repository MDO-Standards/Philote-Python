<div align="center">
<img src="https://github.com/MDO-Standards/Philote-Python/blob/main/doc/graphics/philote-python.svg?raw=true" width="500">
</div>

[//]: # (![Philote]&#40;https://github.com/MDO-Standards/Philote-Python/blob/main/doc/graphics/philote-python.svg?raw=true&#41;)

[![Unit and Integration Tests](https://github.com/MDO-Standards/Philote-Python/actions/workflows/tests.yaml/badge.svg)](https://github.com/MDO-Standards/Philote-Python/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/MDO-Standards/Philote-Python/graph/badge.svg?token=6PK30STMDL)](https://codecov.io/gh/MDO-Standards/Philote-Python)
[![CodeQL](https://github.com/MDO-Standards/Philote-Python/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/MDO-Standards/Philote-Python/actions/workflows/github-code-scanning/codeql)
[![Deploy Documentation](https://github.com/MDO-Standards/Philote-Python/actions/workflows/documentation.yaml/badge.svg)](https://github.com/MDO-Standards/Philote-Python/actions/workflows/documentation.yaml)
![PyPI - Version](https://img.shields.io/pypi/v/philote-mdo?style=flat&color=blue)




# Philote-Python

Python library for using and creating Philote analysis servers.

Documentation can be found at:

https://MDO-Standards.github.io/Philote-Python


## Requirements

The development process requires the following tools to be installed
(they will be installed if not present):

- grpcio-tools
- protoletariat
- importlib.resources

Additionally, the following dependencies are required by Philote MDO and will be
installed automatically during the installation process:

- numpy
- scipy
- grpcio

To run the unit and integration tests, you will need:

- openmdao (can be found [here](https://github.com/OpenMDAO/OpenMDAO) or installed via pip)

## Installation

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


## License

This package is licensed under the Apache 2 license:


>   Copyright 2022-2025 Christopher A. Lupp
>   
>   Licensed under the Apache License, Version 2.0 (the "License");
>   you may not use this file except in compliance with the License.
>   You may obtain a copy of the License at
>   
>       http://www.apache.org/licenses/LICENSE-2.0
>   
>   Unless required by applicable law or agreed to in writing, software
>   distributed under the License is distributed on an "AS IS" BASIS,
>   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
>   See the License for the specific language governing permissions and
>   limitations under the License.



This work has been cleared for public release, distribution unlimited, case
number: AFRL-2023-5713.

The views expressed are those of the authors and do not reflect the official
guidance or position of the United States Government, the Department of Defense
or of the United States Air Force.

Statement from DoD: The Appearance of external hyperlinks does not constitute
endorsement by the United States Department of Defense (DoD) of the linked
websites, of the information, products, or services contained therein. The DoD
does not exercise any editorial, security, or other control over the information
you may find at these locations.

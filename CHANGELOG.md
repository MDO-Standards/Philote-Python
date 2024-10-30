# Change Log

## Version 0.7.0

### Features

- Created a general implementation of the implicit discipline client for
  OpenMDAO. The client creates an OpenMDAO ImplicitComponent which can
  be added to any OpenMDAO model.

### Bug Fixes

- Added a check if OpenMDAO is installed before defining any classes that use
  OpenMDAO types. This is a bug that has never (to my knowledge) been
  encountered, but could force non-OpenMDAO users to install the package, even
  though they have no use for it.
- Fixed grpcio-tools build dependency issue. Under certain circumstances (e.g., use of an older grpcio package), the
  installation will fail due to an incompatible grpcio-tools version getting installed at build time. The grpcio-tools
  version has been fixed for the build at 1.59. As a result the grpcio version also must at least be 1.59


## Version 0.6.1

### Features

- None

### Bug Fixes

- Fixed grpcio-tools build dependency issue. Under certain circumstances (e.g., use of an older grpcio package), the
  installation will fail due to an incompatible grpcio-tools version getting installed at build time. The grpcio-tools
  version has been fixed for the build at 1.59. As a result the grpcio version also must at least be 1.59


## Version 0.6.0

### Features

- Added a mechanism for the server to provide a list of available options
  (with associated types).
- Created a general implementation of the explicit discipline client for
  OpenMDAO. The client creates an OpenMDAO ExplicitComponent which can
  be added to any OpenMDAO model.

### Bug Fixes

- None


## Version 0.5.3

### Features

- None

### Bug Fixes

- Added missing function arguments to explicit discipline.


## Version 0.5.2

### Features

- None

### Bug Fixes

- Lowered the dependency versions (they were far too stringent and new)
- Change PyPI deployment to source only. It is not practical to distribute
  a platform-specific wheel. The wheel must be platform-specific, because gRPC
  has C underpinnings.


## Version 0.5.1

### Features

- Transitioned away from setuptools and setup.py to a pyproject.toml
  and poetry-based package.
- gRPC and protobuf stubs are now automatically compiled during 
  installation.
- Added test coverage report generation that is uploaded to coveralls.
- Added action to upload to PyPI when a release is published.

### Bug Fixes

- Lowered the dependency versions (they were far too stringent and new)
- Change PyPI deployment to source only. It is not practical to distribute
  a platform-specific wheel. The wheel must be platform-specific, because gRPC
  has C underpinnings.


## Version 0.5.0

- yanked due to source distribution issues. All features present in 0.5.1


## Version 0.4.0

### Features

- General documentation updates.

### Bug Fixes

- None


## Version 0.3.0

This release is one of the biggest changes to the code to date. It contains a
fundamental reorganization and adds a number of features. Notably, it adds
unit and integration testing of almost all the code.

### Features

- Reorganized codebase to reduce code duplication. The clients and servers now
  use base classes.
- Protobuf/gRPC files are now generated at build time and not committed
  to the repository. This requires grpc-tools and protoletariat to be installed.
  See the readme for details.
- Added a change log file to the repository.
- Updated API and logic to conform with newer Philote definition.
- Added unit testing suite.
- Added integration test suite (based on examples).
- Completed implicit discipline functionality and testing.
- Fixed unit tests for GetVariableDefinitions and GetPartialsDefinitions.
- Added edge case handling for partials of variables that are scalar.
- 

### Bug Fixes

- Corrected the preallocate_inputs function for the implicit case to resolve
  variable copy issues.
- Fixed typo in discrete input parsing.
- Moved to setup.py, as setuptools is still in beta for pyproject.toml.
- Added jupyter book for documentation.
- Added a quick start guide.


## Version 0.2.1

This is purely a bugfix release. Thanks to Alex Xu for finding these bugs and fixing them.

### Features

- None

### Bug Fixes

- Fixed bug that prevented proper chunking of array data
- Fixed flat view of arrays used during variable transfer


## Version 0.2.0

This version augments the Philote MDO version to 0.3.0.

### Features

- Moved to Philote version 0.3.0
- Renamed RPC function from Compute to Functions for Philote 0.3.0 compatibility
- Renamed RPC function from ComputePartials to Gradient for Philote 0.3.0 compatibility

### Bug Fixes

- Added flattened views for the ndarrays received. The previous version would 
  error for n-dimensional arrays, as the slices would not work unless the array
  was flattened.


## Version 0.1.0

Initial release of the Philote MDO Python bindings. Includes working remote 
explicit disciplines. Only the generic API currently works, so there is no
framework support for OpenMDAO or CSDL.

### Features

- Implemented a remote explicit discipline analysis server API.
- Implemented a corresponding client for explicit analyses.
- Added a simple parabaloid example to demonstrate the server/client in
action.

### Bug Fixes

- None, as this is the first release.

### Note

All versions starting with a 0 as the major version number should be
considered pre-release. While they may work in production environments,
it is expected that bugs may surface and that several features are still
missing. Because of this, the API may still change frequently before version
1.0.0 is released.

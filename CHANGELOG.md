# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Julia discipline wrapper infrastructure for serving pure Julia Philote disciplines via Python gRPC server using juliacall
- JuliaWrapperDiscipline for explicit disciplines
- JuliaImplicitWrapperDiscipline for implicit disciplines
- YAML configuration support via PhiloteConfig for Julia disciplines
- CLI entry point (philote-julia-serve command) for serving Julia disciplines
- Optional 'julia' dependencies group (juliacall and pyyaml)
- Example Julia disciplines (paraboloid, quadratic) with configurations
- Automated release workflow using GitHub Actions
- Copyright update script for Python source files

### Changed

- Convert CHANGELOG to Keep a Changelog format with [Unreleased] section
- CHANGELOG now follows semantic versioning categories (Added/Changed/Fixed/Removed)

## [0.7.0] - 2024-12-18

### Added

- Created a general implementation of the implicit discipline client for OpenMDAO. The client creates an OpenMDAO ImplicitComponent which can be added to any OpenMDAO model.
- Created an interface to host OpenMDAO groups in ExplicitServers.
- Added integration tests for OpenMDAO implicit components using the quadratic example
- Added unit tests for OpenMDAO linearize, solve_nonlinear, and apply_nonlinear functions
- Moved tests out of the package structure

### Changed

- Updated Philote protocol to version 0.7.0
- Improved test coverage (100% lines covered by unit and integration tests, excluding generated files)
- Updated copyright statements across the codebase

### Fixed

- Added a check if OpenMDAO is installed before defining any classes that use OpenMDAO types. This prevents forcing non-OpenMDAO users to install the package.
- Fixed grpcio-tools build dependency issue. Under certain circumstances (e.g., use of an older grpcio package), the installation will fail due to an incompatible grpcio-tools version getting installed at build time. The grpcio-tools version has been fixed for the build at 1.59. As a result the grpcio version also must at least be 1.59

## [0.6.1] - 2024-03-15

### Fixed

- Fixed grpcio-tools build dependency issue. Under certain circumstances (e.g., use of an older grpcio package), the
  installation will fail due to an incompatible grpcio-tools version getting installed at build time. The grpcio-tools
  version has been fixed for the build at 1.59. As a result the grpcio version also must at least be 1.59


## [0.6.0] - 2024-02-01

### Added

- Added a mechanism for the server to provide a list of available options (with associated types).
- Created a general implementation of the explicit discipline client for OpenMDAO. The client creates an OpenMDAO ExplicitComponent which can be added to any OpenMDAO model.

## [0.5.3] - 2023-11-15

### Fixed

- Added missing function arguments to explicit discipline.

## [0.5.2] - 2023-11-10

### Fixed

- Lowered the dependency versions (they were far too stringent and new)
- Change PyPI deployment to source only. It is not practical to distribute a platform-specific wheel. The wheel must be platform-specific, because gRPC has C underpinnings.

## [0.5.1] - 2023-11-05

### Added

- Transitioned away from setuptools and setup.py to a pyproject.toml and poetry-based package.
- gRPC and protobuf stubs are now automatically compiled during installation.
- Added test coverage report generation that is uploaded to coveralls.
- Added action to upload to PyPI when a release is published.

### Fixed

- Lowered the dependency versions (they were far too stringent and new)
- Change PyPI deployment to source only. It is not practical to distribute a platform-specific wheel. The wheel must be platform-specific, because gRPC has C underpinnings.

## [0.5.0] - 2023-11-01

### Removed

- **YANKED**: This version was yanked due to source distribution issues. All features present in 0.5.1

## [0.4.0] - 2023-10-15

### Changed

- General documentation updates.

## [0.3.0] - 2023-08-01

This release is one of the biggest changes to the code to date. It contains a fundamental reorganization and adds a number of features. Notably, it adds unit and integration testing of almost all the code.

### Added

- Reorganized codebase to reduce code duplication. The clients and servers now use base classes.
- Protobuf/gRPC files are now generated at build time and not committed to the repository. This requires grpc-tools and protoletariat to be installed.
- Added a change log file to the repository.
- Added unit testing suite.
- Added integration test suite (based on examples).
- Completed implicit discipline functionality and testing.
- Added edge case handling for partials of variables that are scalar.
- Added jupyter book for documentation.
- Added a quick start guide.

### Changed

- Updated API and logic to conform with newer Philote definition.

### Fixed

- Fixed unit tests for GetVariableDefinitions and GetPartialsDefinitions.
- Corrected the preallocate_inputs function for the implicit case to resolve variable copy issues.
- Fixed typo in discrete input parsing.
- Moved to setup.py, as setuptools is still in beta for pyproject.toml.

## [0.2.1] - 2023-06-15

This is purely a bugfix release. Thanks to Alex Xu for finding these bugs and fixing them.

### Fixed

- Fixed bug that prevented proper chunking of array data
- Fixed flat view of arrays used during variable transfer

## [0.2.0] - 2023-05-01

This version augments the Philote MDO version to 0.3.0.

### Changed

- Moved to Philote version 0.3.0
- Renamed RPC function from Compute to Functions for Philote 0.3.0 compatibility
- Renamed RPC function from ComputePartials to Gradient for Philote 0.3.0 compatibility

### Fixed

- Added flattened views for the ndarrays received. The previous version would error for n-dimensional arrays, as the slices would not work unless the array was flattened.

## [0.1.0] - 2023-03-01

Initial release of the Philote MDO Python bindings. Includes working remote explicit disciplines. Only the generic API currently works, so there is no framework support for OpenMDAO or CSDL.

### Added

- Implemented a remote explicit discipline analysis server API.
- Implemented a corresponding client for explicit analyses.
- Added a simple parabaloid example to demonstrate the server/client in action.

### Note

All versions starting with a 0 as the major version number should be considered pre-release. While they may work in production environments, it is expected that bugs may surface and that several features are still missing. Because of this, the API may still change frequently before version 1.0.0 is released.

[unreleased]: https://github.com/MDO-Standards/Philote-Python/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/MDO-Standards/Philote-Python/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/MDO-Standards/Philote-Python/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/MDO-Standards/Philote-Python/compare/v0.5.3...v0.6.0
[0.5.3]: https://github.com/MDO-Standards/Philote-Python/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/MDO-Standards/Philote-Python/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/MDO-Standards/Philote-Python/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/MDO-Standards/Philote-Python/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/MDO-Standards/Philote-Python/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/MDO-Standards/Philote-Python/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/MDO-Standards/Philote-Python/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/MDO-Standards/Philote-Python/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MDO-Standards/Philote-Python/releases/tag/v0.1.0

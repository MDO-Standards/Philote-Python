# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Features

- Continuous array data is now read and written through the packed wire buffer
  directly, instead of through the protobuf `repeated double` API, which
  converts every element to and from a boxed Python float.  A packed
  `repeated double` is encoded as a length-delimited buffer of little-endian
  doubles, which is byte for byte what NumPy already holds, so the output is
  identical and **the protocol is unchanged** -- peers in any language are
  unaffected.  Moving 200k doubles drops from 36 ms to 0.23 ms per direction.
  End to end, a 1.6 MB call over the unary transport goes from 75 ms to
  3.3 ms, and the same call over an unchunked stream from 71 ms to 2.6 ms.
  The helpers live in `philote_mdo.utils.encoding`.
- `DisciplineClient.unary_max_bytes` now defaults to 256 KiB rather than
  128 KiB.  With the conversion cost removed, the unary transport keeps its
  advantage over a stream up to roughly that size.

- Added a unary compute transport alongside the existing bidirectional
  streaming one.  The standard gains a `VariableSet` message and five unary
  RPCs (`ComputeFunctionUnary`, `ComputeGradientUnary`, `ComputeResidualsUnary`,
  `SolveResidualsUnary`, `ComputeResidualGradientsUnary`), plus
  `supports_unary` and `max_unary_bytes` fields on `DisciplineProperties`.
  Adding methods to the existing services is wire-compatible, so old servers
  simply answer `UNIMPLEMENTED`.
- Clients select the transport automatically (`DisciplineClient.transport`
  defaults to `"auto"`).  The unary transport is used when the payload, sized
  from the variable metadata gathered during setup, fits within
  `unary_max_bytes`; otherwise the streaming transport is used.  A per-call
  size guard catches oversized discrete variables, and a failed unary attempt
  falls back to streaming on `UNIMPLEMENTED` or `RESOURCE_EXHAUSTED`.  Any
  other status propagates rather than being silently retried.  Set
  `transport = "stream"` to pin the previous behaviour, or `"unary"` to force
  the new one.
- Measured with the new `utils/bench_transport.py`, the unary transport cuts
  round-trip latency for small disciplines by 2-6x: a two-scalar discipline
  goes from 486 to 203 microseconds, and a 100-variable one from 10.1 to 1.6
  milliseconds.  Under 16 concurrent clients throughput rises 3.8x and
  per-call client CPU drops 6.7x.  The saving is a fixed per-call cost, so it
  matters in proportion to how often a discipline is called rather than how
  much data it moves.
- Added `philote_mdo.utils.channel_options()` and
  `philote_mdo.utils.server_options()` for sizing gRPC message limits, needed
  when raising `unary_max_bytes` past the 4 MiB default.  These replace the
  `DisciplineClient.grpc_options` attribute, which was never read.
- Added `philote_mdo.utils.get_partials_shape()`, which centralises the
  Jacobian block shape rule that was previously duplicated in
  `DisciplineClient._recover_partials` and
  `DisciplineServer.preallocate_partials`.

### Bug Fixes

- Fixed `ImplicitServer` emitting `Array.end` as an exclusive index, while the
  explicit server and both clients treat it as inclusive per the standard.  Any
  implicit variable spanning more than one chunk failed to decode; single-chunk
  arrays were masked by NumPy slice clipping (#66).
- Fixed the `GetInfo` RPC, which was implemented as a generator on the server
  and indexed as a stream on the client, even though the proto declares it as
  unary.  Every call failed with `Failed to serialize response!`.  The server
  now returns a `DisciplineProperties` message and the client reads it
  directly (#67).
- The server now populates the `name` and `version` fields of
  `DisciplineProperties` from the discipline's `_name` / `_version`
  attributes, and the client stores them (#67).

## [0.8.0] - 2026-05-16

### Features

- Added `"double"` and `"string"` as type aliases in the OpenMDAO type map,
  mapping them to `float` and `str` respectively, to match common proto type
  name conventions.
- Added dynamic shapes for inputs and outputs.  Disciplines can now
  declare variables with `dynamic_shape=True` in `add_input` /
  `add_output`, indicating that the client is allowed to set the
  variable's shape at runtime.  A new `SetVariableShapes` gRPC RPC
  lets clients send resolved shapes after querying variable definitions.
  The OpenMDAO bindings automatically map dynamic-shape variables to
  `shape_by_conn=True` and send resolved shapes back to the server
  (MDO-Standards/Philote-MDO#6).
- Added support for struct (dict) options via the new `kStruct` DataType enum
  value, enabling complex nested data to be declared and passed as discipline
  options (#49).
- Added discrete variable support throughout the stack.  Disciplines can
  now declare discrete inputs/outputs via `add_discrete_input` /
  `add_discrete_output`.  Discrete data is serialized as
  `google.protobuf.Value` (supporting scalars, lists, and nested
  structures) and multiplexed alongside continuous `Array` chunks in
  the new `VariableMessage` wrapper.  The OpenMDAO bindings
  (`RemoteExplicitComponent`, `RemoteImplicitComponent`) automatically
  discover and forward discrete variables.
- Added comprehensive input validation and error handling across the
  framework.  Introduces custom exception classes (`PhiloteValidationError`,
  `PhiloteServerError`), parameter validation in discipline base classes
  (`add_input`, `add_output`, `add_option`, `declare_partials`), proper
  gRPC error propagation via `context.abort()` with appropriate status
  codes in all server RPC methods, and client-side input validation with
  gRPC error wrapping (#46).

### Bug Fixes

- Fixed bare `except` to `except ImportError` in `examples/__init__.py`.
- Fixed missing space in `RemoteImplicitComponent` error message
  ("will notbe" -> "will not be").
- Fixed `SellarMDA` promoted-input ambiguity that newer OpenMDAO releases
  reject during `final_setup`. The `x` and `z` defaults were being set on
  the inner `cycle` subgroup, but `obj_cmp` promoted the same variables
  to the top level with different defaults. Moved the `set_input_defaults`
  calls to the top-level `SellarMDA` group so the promoted values agree.
- Updated `test_implicit_discipline_apply_linear_not_implemented` to call
  `apply_linear` with its current six-argument signature (`inputs`,
  `outputs`, `d_inputs`, `d_outputs`, `d_residuals`, `mode`); the stale
  three-argument call raised `TypeError` before the `NotImplementedError`
  assertion could fire.

### Documentation & Infrastructure

- Added Codecov configuration (`codecov.yml`) requiring 95% coverage on
  both project total and patch (new/changed lines).
- Added `fail_under = 95` to `.coveragerc` for local coverage enforcement.
- Marked unreachable import guards and defensive branches with
  `pragma: no cover`.
- Updated installation instructions to reflect PyPI install option.
- Added documentation for implicit disciplines.
- Added documentation for OpenMDAO clients
- Added documentation for the OpenMDOA subproblem discipline.
- Migrated documentation from Jupyter Book to Docusaurus, mirroring the
  Delphi docs setup (dark-mode theme, KaTeX math, custom landing page,
  versioned docs).
- Replaced the Jupyter Book `gh-pages` deploy job with a Docusaurus
  GitHub Pages artifact pipeline triggered on `develop` for `docs/**`.
- Added a `Release` GitHub Actions workflow mirroring Delphi's release
  flow: PR-label-driven version bumps, signed commits, `pyproject.toml`
  version update, CHANGELOG rewrite, Docusaurus version snapshots on
  stable releases, GitHub Release creation, and auto-merge of `main`
  back to `develop`.
- Reformatted `CHANGELOG.md` to the Keep a Changelog convention so the
  release workflow can drive header rewrites and comparison links.

## [0.7.0]

### Features

- Created a general implementation of the implicit discipline client for
  OpenMDAO. The client creates an OpenMDAO ImplicitComponent which can
  be added to any OpenMDAO model.
- Created an interface to host OpenMDAO groups in ExplicitServers.
- Added integration tests for OpenMDAO implicit components using the quadratic example
- Added unit tests for OpenMDAO linearize, solve_nonlinear, and apply_nonlinear functions
- Updated Philote protocol to version 0.7.0
- Moved tests out of the package structure
- Improved test coverage (100% lines covered by unit and integration tests, excluding generated files)

### Bug Fixes

- Added a check if OpenMDAO is installed before defining any classes that use
  OpenMDAO types. This is a bug that has never (to my knowledge) been
  encountered, but could force non-OpenMDAO users to install the package, even
  though they have no use for it.
- Fixed grpcio-tools build dependency issue. Under certain circumstances (e.g., use of an older grpcio package), the
  installation will fail due to an incompatible grpcio-tools version getting installed at build time. The grpcio-tools
  version has been fixed for the build at 1.59. As a result the grpcio version also must at least be 1.59

### Documentation & Infrastructure

- Updated copyright statements across the codebase

## [0.6.1]

### Features

- None

### Bug Fixes

- Fixed grpcio-tools build dependency issue. Under certain circumstances (e.g., use of an older grpcio package), the
  installation will fail due to an incompatible grpcio-tools version getting installed at build time. The grpcio-tools
  version has been fixed for the build at 1.59. As a result the grpcio version also must at least be 1.59

## [0.6.0]

### Features

- Added a mechanism for the server to provide a list of available options
  (with associated types).
- Created a general implementation of the explicit discipline client for
  OpenMDAO. The client creates an OpenMDAO ExplicitComponent which can
  be added to any OpenMDAO model.

### Bug Fixes

- None

## [0.5.3]

### Features

- None

### Bug Fixes

- Added missing function arguments to explicit discipline.

## [0.5.2]

### Features

- None

### Bug Fixes

- Lowered the dependency versions (they were far too stringent and new)
- Change PyPI deployment to source only. It is not practical to distribute
  a platform-specific wheel. The wheel must be platform-specific, because gRPC
  has C underpinnings.

## [0.5.1]

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

## [0.5.0]

- yanked due to source distribution issues. All features present in 0.5.1

## [0.4.0]

### Features

- General documentation updates.

### Bug Fixes

- None

## [0.3.0]

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

### Bug Fixes

- Corrected the preallocate_inputs function for the implicit case to resolve
  variable copy issues.
- Fixed typo in discrete input parsing.
- Moved to setup.py, as setuptools is still in beta for pyproject.toml.
- Added jupyter book for documentation.
- Added a quick start guide.

## [0.2.1]

This is purely a bugfix release. Thanks to Alex Xu for finding these bugs and fixing them.

### Features

- None

### Bug Fixes

- Fixed bug that prevented proper chunking of array data
- Fixed flat view of arrays used during variable transfer

## [0.2.0]

This version augments the Philote MDO version to 0.3.0.

### Features

- Moved to Philote version 0.3.0
- Renamed RPC function from Compute to Functions for Philote 0.3.0 compatibility
- Renamed RPC function from ComputePartials to Gradient for Philote 0.3.0 compatibility

### Bug Fixes

- Added flattened views for the ndarrays received. The previous version would 
  error for n-dimensional arrays, as the slices would not work unless the array
  was flattened.

## [0.1.0]

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

[0.7.0]: https://github.com/MDO-Standards/Philote-Python/releases/tag/v0.7.0
[0.6.1]: https://github.com/MDO-Standards/Philote-Python/releases/tag/v0.6.1
[0.6.0]: https://github.com/MDO-Standards/Philote-Python/releases/tag/v0.6.0
[0.5.3]: https://github.com/MDO-Standards/Philote-Python/releases/tag/v0.5.3
[0.5.2]: https://github.com/MDO-Standards/Philote-Python/releases/tag/v0.5.2
[0.5.1]: https://github.com/MDO-Standards/Philote-Python/releases/tag/v0.5.1
[0.5.0]: https://github.com/MDO-Standards/Philote-Python/releases/tag/v0.5.0
[0.4.0]: https://github.com/MDO-Standards/Philote-Python/releases/tag/v0.4.0
[0.3.0]: https://github.com/MDO-Standards/Philote-Python/releases/tag/v0.3.0
[0.2.1]: https://github.com/MDO-Standards/Philote-Python/releases/tag/v0.2.1
[0.2.0]: https://github.com/MDO-Standards/Philote-Python/releases/tag/v0.2.0
[0.1.0]: https://github.com/MDO-Standards/Philote-Python/releases/tag/v0.1.0
[Unreleased]: https://github.com/MDO-Standards/Philote-Python/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/MDO-Standards/Philote-Python/releases/tag/v0.8.0

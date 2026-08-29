# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Features

- Continuous array data is now read and written through the packed wire
  buffer directly, rather than through the protobuf `repeated double` API,
  which converts every element to and from a boxed Python float.  A packed
  `repeated double` is encoded as a length-delimited buffer of little-endian
  doubles, which is byte for byte what NumPy already holds, so the emitted
  bytes are identical and **the protocol is unchanged** -- peers in any
  language are unaffected.  Per direction for 200k doubles, encoding drops
  from 27.4 ms to 0.15 ms and decoding from 7.9 ms to 0.11 ms; a 250k-element
  round trip goes from 121 ms to 36 ms at the 1,000-double chunk size that was
  the default at the time, which the chunking change below takes further.
  Decoding stays on the protobuf container below 64 elements, where
  re-serializing the message to reach its payload would cost more than it
  saves, so scalar-heavy disciplines are unaffected.  The helpers live in
  `philote_mdo.utils.encoding`.
- `DisciplineServer.preallocate_partials` and
  `DisciplineClient._recover_partials` rescanned the full variable metadata
  list twice per declared partial, which is quadratic in the number of
  variables and was paid on every gradient call.  Both now index the metadata
  by name once.  For a discipline with 100 variables and 100 partials,
  `preallocate_partials` drops from 4.1 ms to 0.16 ms and a full gradient
  round trip from 23.0 ms to 14.6 ms.  The Jacobian block shape rule the two
  sites duplicated is now `philote_mdo.utils.get_partials_shape()`.
- `get_chunk_indices` now returns the single-chunk case directly instead of
  deriving it through two NumPy array constructions.  Every variable that fits
  in one chunk takes this path, which is most of them for a discipline of
  scalars: 0.095 ms to 0.005 ms across 100 variables.
- The default `num_double` stream option rises from 1,000 to 100,000 on both
  the client and the server.  With the encoding cost gone, a stream's runtime
  is set by how many messages it carries rather than by how large they are,
  at roughly 64 microseconds per message per direction.  A 250k-element round
  trip drops from 41.7 ms to 2.9 ms.  A chunk is about 780 KiB at the new
  default, against gRPC's 4 MiB message ceiling.  This is a default only:
  `StreamOptions` is negotiated as before, and anyone who sets `num_double`
  explicitly is unaffected.
- `add_input`, `add_output`, `add_discrete_input` and `add_discrete_output`
  checked for a duplicate declaration by scanning the whole metadata list,
  making the cost of declaring a discipline quadratic in the number of
  variables.  They now consult a set of the declared `(type, name)` pairs.
  Declaring 2,000 variables drops from 684 ms to 6.7 ms, and the cost is now
  linear.
- `SetVariableShapes` on the server and `send_variable_shapes` on the client
  resolved each incoming shape by scanning the whole variable metadata list,
  twice for an implicit output, which is quadratic in the number of dynamic
  variables.  Both now index the metadata by type and name once.  Applying
  shapes to 1,000 dynamic variables drops from 922 ms to 120 ms.

### Bug Fixes

- Fixed `RemoteExplicitComponent` and `RemoteImplicitComponent` sending the raw
  constructor keyword arguments to the server rather than the component's
  resolved options.  An option that reached the component by any other route --
  an OpenMDAO default, or an assignment such as `comp.options['dimension'] = 10`
  after construction -- never reached the server, which kept computing with
  whatever it had.  The failure was silent: the component reported the option as
  set while the server used a different value.  The options are now read from
  `comp.options`, restricted to the names the server declared, and transmitted
  during `setup()` immediately before the remote `Setup` call, so post-
  construction assignment takes effect.  Options declared without a value are
  skipped, leaving the server on its own default (#77).
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
- `DisciplineClient.get_partials_definitions` guarded against duplicates by
  testing a `str` against a list of `PartialsMetaData` messages, so the guard
  never fired and every message was appended unconditionally.  Both
  `get_variable_definitions` and `get_partials_definitions` now clear their
  metadata lists before repopulating them, mirroring the `_clear_data()` the
  server performs at the start of each `Setup`, so a client that is set up
  more than once replaces its metadata rather than accumulating duplicate
  Jacobian preallocations and `declare_partials` calls (#78).

### Documentation & Infrastructure

- Updated the documentation site's dependencies to clear known npm
  advisories, taking the audit from 42 findings (2 critical, 26 high) to 17
  (all high).  `npm audit fix` resolved the `webpack-dev-server`, `sockjs`,
  `ws` and `websocket-driver` chains, and `overrides` pin
  `serialize-javascript` to `^7.1.0` and `uuid` to `^11.1.1`, neither of
  which had a fix path through their parents.  The 17 that remain are all
  `image-size`, reached through `@docusaurus/mdx-loader`, whose advisory
  covers every published version, so no upgrade resolves it.  All of these
  are build-time only: the deployed site is static, so they affect a
  developer running `npm run start` and the CI build, not readers of the
  docs.

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

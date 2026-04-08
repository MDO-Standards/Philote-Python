# AI Agent Guidelines for Project Philote-Python

## Project Overview

Philote-Python is the reference Python implementation of the Philote-MDO standard — a gRPC-based protocol for distributed multidisciplinary analysis and optimization. It provides explicit and implicit discipline base classes, the corresponding servers and clients, and OpenMDAO bindings (`RemoteExplicitComponent`, `RemoteImplicitComponent`, `OpenMdaoSubProblem`).

## Build System

### Requirements
- Python 3.9 - 3.12
- `numpy`, `scipy`, `grpcio`, `protobuf` (installed automatically)
- `openmdao` (only for the OpenMDAO bindings and integration tests)

### Install
```bash
# From PyPI
pip install philote-mdo

# From source (editable)
pip install -e .
```

### Regenerating gRPC stubs
Generated `*_pb2.py` / `*_pb2_grpc.py` files are committed to the repo. Regenerate only when the `.proto` definitions change:
```bash
python utils/compile_proto.py
```

### Running tests
```bash
pytest tests/
```
Some tests require OpenMDAO; install it first if you want full coverage.

## Code Organization

```
philote_mdo/
├── general/        # Core discipline / client / server base classes
├── openmdao/       # OpenMDAO bindings (RemoteExplicit/Implicit, OpenMdaoSubProblem)
├── examples/       # Reference disciplines (Paraboloid, Quadratic, Sellar, Rosenbrock)
├── generated/      # Auto-generated gRPC stubs (do not edit by hand)
└── utils/          # Shared utilities

tests/              # pytest unit and integration tests (lives outside the package)
proto/              # Philote-MDO .proto definitions
utils/              # Developer scripts (e.g. compile_proto.py)
docs/               # Docusaurus documentation site (see docs/README.md)
```

## CI/CD

### Tests workflow (`.github/workflows/tests.yaml`)
Runs `pytest` against Python 3.9 / 3.10 / 3.11 / 3.12 on every push to `main`, `develop`, `release/*`, `support/*`, and on PRs into `main` / `develop`.

### Documentation workflow (`.github/workflows/documentation.yaml`)
Builds the Docusaurus site under `docs/` and deploys it to GitHub Pages whenever `docs/**` changes on `develop`. Uses the GitHub Pages artifact pipeline (`actions/configure-pages` + `upload-pages-artifact` + `deploy-pages`).

### PyPI publish workflow (`.github/workflows/publish-pypi.yaml`)
Triggered when a GitHub Release is published. Builds the source distribution with Poetry and pushes to PyPI via trusted publishing. **The release workflow below is what creates the GitHub Release that triggers this.**

### Release workflow (`.github/workflows/release.yaml`)

Label-driven release pipeline mirroring Delphi. Triggered when a labeled PR is **merged** into `main`.

**Label system** (applied to PRs targeting `main`):

| Purpose | Labels (pick exactly one per row) |
|---------|-----------------------------------|
| Release type | `release` OR `prerelease` |
| Version bump | `major`, `minor`, or `patch` |
| Pre-release qualifier | `alpha`, `beta`, or `rc` (only with `prerelease`) |

**What the workflow does on merge:**
1. Generates an ephemeral GPG key and configures signed commits/tags as `github-actions[bot]`.
2. Validates the PR labels (exactly one release type, one bump, optional qualifier on prereleases).
3. Reads the current version from `pyproject.toml` and bumps it per the labels.
   - For prereleases, scans existing `vX.Y.Z-<qual>.*` tags and increments the suffix counter.
4. Updates `pyproject.toml` (`version = "X.Y.Z"`).
5. Updates `CHANGELOG.md`:
   - Replaces `## [Unreleased]` with `## [Unreleased]\n\n## [X.Y.Z] - YYYY-MM-DD`.
   - Removes the old `[Unreleased]: ...` link and appends new comparison/release links.
6. **Stable releases only** — cuts a Docusaurus version snapshot:
   - `npm ci` in `docs/`
   - `npx docusaurus docs:version <X.Y.Z>` (creates `docs/versioned_docs/version-<X.Y.Z>/`, `docs/versioned_sidebars/`, and updates `docs/versions.json`)
   - `sed -i 's/lastVersion: "[^"]*"/lastVersion: "<X.Y.Z>"/'` in `docs/docusaurus.config.ts`
7. Creates a signed commit `chore: release version X.Y.Z` and an annotated tag `vX.Y.Z` (or `vX.Y.Z-<qual>.<n>` for prereleases).
8. Pushes the commit and the tag to `main`.
9. Extracts release notes for the new version from `CHANGELOG.md` via `awk`.
10. Creates a GitHub Release (`softprops/action-gh-release@v2`) with `prerelease: true|false` derived from the labels — this is what triggers `publish-pypi.yaml`.
11. Merges `main` back into `develop` automatically. On merge conflict, opens a PR instead.

**Required setup** (one-time, in repo settings):
- A PAT with `repo` + `workflow` scope stored as the `RELEASE_TOKEN` Actions secret.
- The PR labels listed above must exist on the repo.
- GitHub Pages source must be set to "GitHub Actions".

### Changelog
`CHANGELOG.md` follows the [Keep a Changelog](https://keepachangelog.com/) format. New changes go under `[Unreleased]`. The release workflow converts this header to a versioned section automatically.

**Important**: When adding, changing, or removing features, always update `CHANGELOG.md` as part of the same commit. Add a concise entry under the appropriate subsection of `[Unreleased]`:
- `### Features` — new features or capabilities
- `### Bug Fixes` — bug fixes
- `### Documentation & Infrastructure` — docs, CI, build, and tooling changes

## Documentation

Documentation lives in `docs/` and is built with [Docusaurus](https://docusaurus.io/) (mirrors the Delphi setup: dark-mode default, KaTeX math, custom landing page, versioned docs).

### Local development
```bash
cd docs
npm install        # first time only
npm run start      # live-reload dev server
npm run build      # production build into docs/build
```

### Adding or editing pages
- Source files live under `docs/docs/` and are organised by sidebar category (`getting-started/`, `tutorials/`, `openmdao/`, `about/`).
- The sidebar layout is defined in `docs/sidebars.ts`. Add new pages there if you want them to appear in the nav.
- Math uses KaTeX (`$...$` inline, `$$...$$` block).
- Admonitions use Docusaurus syntax (`:::note`, `:::warning`, `:::tip`, …).
- Edits to `docs/**` on `develop` trigger an automatic deploy.

### Versioned docs
- The `docs/docs/` directory is the **"Next"** branch — edits here feed the in-progress documentation.
- `lastVersion` in `docs/docusaurus.config.ts` is updated automatically by the release workflow.
- **Do not run `npx docusaurus docs:version` manually.** The release workflow cuts the snapshot for every stable release. Manual snapshots desync the version dropdown from the actual git tags.

## Pull Requests

When creating PRs:
- Target `develop` for feature/bugfix work, `main` only for releases.
- Add a concise `[Unreleased]` entry in `CHANGELOG.md` as part of the same PR.
- For release PRs, add the labels documented in the [Release workflow](#release-workflow-githubworkflowsreleaseyaml) section above.

## Git Workflow

- **`main`** — stable releases only. Moves only via the release workflow.
- **`develop`** — integration branch. Feature branches merge here.
- **`feature/*`** — feature branches off `develop`.
- After a release the workflow merges `main` back into `develop` automatically (PR fallback on conflict).

## Creating a Release

1. Make sure `CHANGELOG.md` has meaningful entries under `[Unreleased]`.
2. Open a PR from `develop` → `main`.
3. Apply labels:
   - One of `release` or `prerelease`.
   - One of `major`, `minor`, or `patch`.
   - For prereleases, also one of `alpha`, `beta`, or `rc`.
4. Merge the PR. The release workflow does everything else: version bump, changelog rewrite, docs version snapshot (stable releases), signed commit + tag, GitHub Release, PyPI publish (via the published-release trigger), and merge-back into `develop`.

## License

Apache-2.0. See `LICENSE` and `docs/docs/about/license.md`.

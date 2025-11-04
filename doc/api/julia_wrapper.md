# Julia Wrapper API Reference

This page documents the Python API for working with Julia disciplines in Philote-Python.

## Wrapper Classes

### JuliaWrapperDiscipline

Python discipline wrapper for explicit Julia disciplines.

```python
class JuliaWrapperDiscipline(ExplicitDiscipline)
```

This class loads and executes Julia code via `juliacall`, presenting a pure Python interface compatible with the Philote-Python server infrastructure.

#### Constructor

```python
def __init__(self, julia_file, julia_type, options=None)
```

**Parameters:**
- `julia_file` (str): Path to Julia file containing the discipline (absolute or relative)
- `julia_type` (str): Name of the Julia struct to instantiate (e.g., `"ParaboloidDiscipline"`)
- `options` (dict, optional): Dictionary of discipline options to set after initialization

**Raises:**
- `FileNotFoundError`: If `julia_file` does not exist
- `ValueError`: If `julia_type` cannot be instantiated
- `ImportError`: If `juliacall` is not installed

**Example:**

```python
from philote_mdo.wrappers.julia import JuliaWrapperDiscipline

discipline = JuliaWrapperDiscipline(
    julia_file="/path/to/paraboloid.jl",
    julia_type="ParaboloidDiscipline",
    options={"scale_factor": 2.0}
)
```

#### Methods

##### setup()

```python
def setup(self)
```

Define inputs, outputs, and partials based on Julia discipline metadata.

This method reads metadata from the Julia discipline (automatically populated during `setup!()`) and configures the Python discipline interface accordingly.

**Called automatically by the server - users typically don't call this directly.**

##### compute(inputs, outputs)

```python
def compute(self, inputs, outputs)
```

Compute outputs from inputs by calling the Julia discipline's `compute()` function.

**Parameters:**
- `inputs` (dict): Dictionary mapping input names to NumPy arrays
- `outputs` (dict): Dictionary to populate with output arrays

**Raises:**
- `RuntimeError`: If the Julia `compute()` function fails

##### compute_partials(inputs, partials)

```python
def compute_partials(self, inputs, partials)
```

Compute partial derivatives by calling the Julia discipline's `compute_partials()` function.

**Parameters:**
- `inputs` (dict): Dictionary mapping input names to NumPy arrays
- `partials` (dict): Dictionary to populate with Jacobian arrays (keyed by `(output_name, input_name)` tuples)

**Raises:**
- `RuntimeError`: If the Julia `compute_partials()` function fails

---

### JuliaImplicitWrapperDiscipline

Python discipline wrapper for implicit Julia disciplines.

```python
class JuliaImplicitWrapperDiscipline(ImplicitDiscipline)
```

This class loads and executes Julia implicit disciplines via `juliacall`, supporting residual-based formulations.

#### Constructor

```python
def __init__(self, julia_file, julia_type, options=None)
```

**Parameters:**
- `julia_file` (str): Path to Julia file containing the implicit discipline
- `julia_type` (str): Name of the Julia struct to instantiate
- `options` (dict, optional): Dictionary of discipline options

**Raises:**
- `FileNotFoundError`: If `julia_file` does not exist
- `ValueError`: If `julia_type` cannot be instantiated
- `ImportError`: If `juliacall` is not installed

**Example:**

```python
from philote_mdo.wrappers.julia import JuliaImplicitWrapperDiscipline

discipline = JuliaImplicitWrapperDiscipline(
    julia_file="/path/to/quadratic.jl",
    julia_type="QuadraticDiscipline"
)
```

#### Methods

##### setup()

```python
def setup(self)
```

Define inputs, outputs, residuals, and partials based on Julia discipline metadata.

**Called automatically by the server.**

##### compute_residuals(inputs, outputs, residuals)

```python
def compute_residuals(self, inputs, outputs, residuals)
```

Compute residuals by calling the Julia discipline's `compute_residuals()` function.

**Parameters:**
- `inputs` (dict): Dictionary mapping input names to NumPy arrays
- `outputs` (dict): Dictionary mapping output names to NumPy arrays
- `residuals` (dict): Dictionary to populate with residual arrays

**Raises:**
- `RuntimeError`: If the Julia `compute_residuals()` function fails

##### solve_residuals(inputs, outputs)

```python
def solve_residuals(self, inputs, outputs)
```

Solve for outputs that drive residuals to zero by calling the Julia discipline's `solve_residuals()` function.

**Parameters:**
- `inputs` (dict): Dictionary mapping input names to NumPy arrays
- `outputs` (dict): Dictionary mapping output names to NumPy arrays (modified in place)

**Raises:**
- `RuntimeError`: If the Julia `solve_residuals()` function fails

##### residual_partials(inputs, outputs, partials)

```python
def residual_partials(self, inputs, outputs, partials)
```

Compute residual partial derivatives by calling the Julia discipline's `residual_partials()` function.

**Parameters:**
- `inputs` (dict): Dictionary mapping input names to NumPy arrays
- `outputs` (dict): Dictionary mapping output names to NumPy arrays
- `partials` (dict): Dictionary to populate with Jacobian arrays (keyed by `(residual_name, variable_name)` tuples)

**Raises:**
- `RuntimeError`: If the Julia `residual_partials()` function fails

---

## Server Functions

### serve_explicit_discipline

```python
def serve_explicit_discipline(config: PhiloteConfig)
```

Start a gRPC server hosting an explicit Julia discipline.

**Parameters:**
- `config` (PhiloteConfig): Configuration object with discipline and server settings

**Example:**

```python
from philote_mdo.wrappers.julia import serve_explicit_discipline, PhiloteConfig

config = PhiloteConfig.from_yaml("config.yaml")
serve_explicit_discipline(config)
```

This function:
1. Creates a `JuliaWrapperDiscipline` from the configuration
2. Creates a gRPC server with the specified settings
3. Attaches the discipline to the server
4. Starts the server and waits for termination

**Blocks until server is stopped (Ctrl+C).**

### serve_implicit_discipline

```python
def serve_implicit_discipline(config: PhiloteConfig)
```

Start a gRPC server hosting an implicit Julia discipline.

**Parameters:**
- `config` (PhiloteConfig): Configuration object with discipline and server settings

**Example:**

```python
from philote_mdo.wrappers.julia import serve_implicit_discipline, PhiloteConfig

config = PhiloteConfig.from_yaml("quadratic_config.yaml")
serve_implicit_discipline(config)
```

**Blocks until server is stopped (Ctrl+C).**

---

## Configuration Classes

### PhiloteConfig

Complete configuration for a Philote-Julia server.

```python
@dataclass
class PhiloteConfig:
    discipline: DisciplineConfig
    server: ServerConfig
```

#### Class Methods

##### from_yaml(yaml_path)

```python
@classmethod
def from_yaml(cls, yaml_path: str) -> PhiloteConfig
```

Load configuration from a YAML file.

**Parameters:**
- `yaml_path` (str): Path to YAML configuration file

**Returns:**
- `PhiloteConfig`: Configuration object

**Raises:**
- `FileNotFoundError`: If configuration file or Julia file doesn't exist
- `ValueError`: If configuration is invalid

**Example:**

```python
config = PhiloteConfig.from_yaml("/path/to/config.yaml")
```

**Note:** Relative paths in `julia_file` are resolved relative to the YAML file's directory.

##### to_yaml(yaml_path)

```python
def to_yaml(self, yaml_path: str)
```

Write configuration to a YAML file.

**Parameters:**
- `yaml_path` (str): Path to write YAML configuration

**Example:**

```python
config.to_yaml("output_config.yaml")
```

---

### DisciplineConfig

Configuration for a Julia discipline.

```python
@dataclass
class DisciplineConfig:
    kind: str              # "explicit" or "implicit"
    julia_file: str        # Path to .jl file
    julia_type: str        # Julia struct name
    options: Dict[str, any] = field(default_factory=dict)
```

**Validation:**
- `kind` must be `"explicit"` or `"implicit"`
- `julia_file` and `julia_type` are required

**Example:**

```python
from philote_mdo.wrappers.julia import DisciplineConfig

disc_config = DisciplineConfig(
    kind="explicit",
    julia_file="paraboloid.jl",
    julia_type="ParaboloidDiscipline",
    options={"scale_factor": 2.0}
)
```

---

### ServerConfig

Configuration for the gRPC server.

```python
@dataclass
class ServerConfig:
    address: str = "[::]:50051"
    max_workers: int = 10
```

**Validation:**
- `max_workers` must be >= 1

**Example:**

```python
from philote_mdo.wrappers.julia import ServerConfig

server_config = ServerConfig(
    address="localhost:50052",
    max_workers=4
)
```

---

## Command-Line Interface

### philote-julia-serve

Command-line tool for serving Julia disciplines.

```bash
philote-julia-serve <config.yaml>
```

**Arguments:**
- `config.yaml`: Path to YAML configuration file

**Example:**

```bash
philote-julia-serve paraboloid_config.yaml
```

**Output:**

```
======================================================================
  Philote Julia Server (Python wrapper + juliacall)
======================================================================

Configuration:
  Julia file:  /path/to/paraboloid.jl
  Julia type:  ParaboloidDiscipline
  Server addr: [::]:50051
  Max workers: 10

Loading Julia discipline from: /path/to/paraboloid.jl
✓ Julia discipline loaded: ParaboloidDiscipline

✓ Server started successfully!
  Listening on: [::]:50051

Press Ctrl+C to stop the server.
======================================================================
```

**Implementation:**

The CLI is defined in `philote_mdo/wrappers/julia/cli.py` and registered as a console script in `pyproject.toml`:

```toml
[project.scripts]
philote-julia-serve = "philote_mdo.wrappers.julia.cli:main"
```

---

## Usage Patterns

### Creating a Server Programmatically

Instead of using the CLI, you can create servers programmatically:

```python
from concurrent import futures
import grpc
from philote_mdo.wrappers.julia import JuliaWrapperDiscipline
import philote_mdo.general as pmdo

# Create the wrapper discipline
discipline_wrapper = JuliaWrapperDiscipline(
    julia_file="paraboloid.jl",
    julia_type="ParaboloidDiscipline",
    options={"scale_factor": 2.0}
)

# Create gRPC server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

# Attach discipline to server
discipline_server = pmdo.ExplicitServer(discipline=discipline_wrapper)
discipline_server.attach_to_server(server)

# Start server
server.add_insecure_port("[::]:50051")
server.start()
print("Server running...")

# Wait for termination
try:
    server.wait_for_termination()
except KeyboardInterrupt:
    server.stop(grace=2.0)
```

### Using with Context Managers

For testing or temporary servers:

```python
import grpc
from philote_mdo.general import RemoteExplicitDiscipline

# Connect to Julia discipline server
with grpc.insecure_channel('localhost:50051') as channel:
    discipline = RemoteExplicitDiscipline(channel)

    # Use the discipline
    outputs = discipline.compute({'x': [1.0], 'y': [2.0]})
    print(outputs)
```

### Batch Processing

Serve multiple Julia disciplines on different ports:

```python
import subprocess
import threading

def serve_discipline(config_file):
    subprocess.run(['philote-julia-serve', config_file])

# Start multiple servers in parallel
threads = [
    threading.Thread(target=serve_discipline, args=('config1.yaml',)),
    threading.Thread(target=serve_discipline, args=('config2.yaml',)),
]

for t in threads:
    t.start()

for t in threads:
    t.join()
```

---

## Error Handling

Common errors and their meanings:

### FileNotFoundError

```python
FileNotFoundError: Julia file not found: /path/to/file.jl
```

**Cause:** The specified `.jl` file does not exist.

**Solution:** Check the path in your configuration file. Ensure the file exists and is readable.

### ValueError: Failed to instantiate Julia type

```python
ValueError: Failed to instantiate Julia type 'ParaboloidDiscipline'
```

**Cause:** The Julia type name doesn't exist in the loaded file.

**Solution:**
- Verify the struct name matches exactly (case-sensitive)
- Ensure the struct is defined in the `.jl` file
- Check for typos in the configuration

### RuntimeError: Error in Julia compute

```python
RuntimeError: Error in Julia compute: MethodError(...)
```

**Cause:** The Julia `compute()` function raised an error.

**Solution:**
- Check Julia function implementation for bugs
- Verify input types and shapes match expectations
- Review error message for Julia-specific details

### ImportError: Cannot import juliacall

```python
ImportError: Cannot import juliacall
```

**Cause:** The `juliacall` package is not installed.

**Solution:** Install Julia dependencies:
```bash
pip install philote-mdo[julia]
```

---

## See Also

- {ref}`tutorials:julia` - Julia integration tutorial
- {ref}`tutorials:explicit` - Creating explicit disciplines
- {ref}`tutorials:implicit` - Creating implicit disciplines
- [Philote.jl Repository](https://github.com/MDO-Standards/Philote-Julia) - Julia package documentation

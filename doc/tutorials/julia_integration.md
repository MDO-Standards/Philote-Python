(tutorials:julia)=
# Julia Integration

Philote-Python supports serving disciplines written in pure Julia via the Philote.jl package. This integration enables Julia developers to leverage Julia's high-performance numerical computing capabilities while using Python's proven gRPC server infrastructure.

:::{note}
This guide assumes you have basic familiarity with both Julia and the Philote discipline concept. If you're new to Philote, start with the {ref}`tutorials:quick_start` guide.
:::

## Overview

The Julia integration uses a bridge architecture:

1. **Julia developers** write disciplines using the [Philote.jl](https://github.com/MDO-Standards/Philote-Julia) module
2. **Python wrapper classes** in Philote-Python load and execute Julia code via `juliacall`
3. **Python gRPC servers** serve these Julia disciplines to any Philote client

This approach combines Julia's computational performance with Python's mature gRPC infrastructure, enabling zero-copy data transfer between Python and Julia.

## Installation

To use Julia disciplines with Philote-Python, you need to install the Julia extra dependencies:

```bash
pip install philote-mdo[julia]
```

This installs the required dependencies:
- `juliacall` - Python-Julia bridge for zero-copy interop
- `pyyaml` - YAML configuration file parsing

You'll also need the Philote.jl Julia package. The wrapper will automatically load it from your Julia environment, or you can install it manually:

```julia
using Pkg
Pkg.add(url="https://github.com/MDO-Standards/Philote-Julia")
```

## Creating Julia Disciplines

Julia disciplines are created by defining a struct that inherits from one of Philote.jl's abstract types and implementing the required interface methods.

### Explicit Disciplines

Explicit disciplines compute outputs directly from inputs: `outputs = f(inputs)`.

Here's a simple example implementing the paraboloid function:

\begin{align}
f(x,y) &= (x-3)^2 + x y + (y+4)^2 - 3
\end{align}

```julia
using Philote

# Define a struct that inherits from ExplicitDiscipline
mutable struct ParaboloidDiscipline <: Philote.ExplicitDiscipline
    scale_factor::Float64
    offset::Float64

    function ParaboloidDiscipline()
        new(1.0, 0.0)
    end
end

# Declare inputs, outputs, options, and partials
function Philote.setup!(discipline::ParaboloidDiscipline)
    # Declare options
    Philote.add_option!(discipline, "scale_factor", "float")
    Philote.add_option!(discipline, "offset", "float")

    # Declare inputs
    Philote.add_input!(discipline, "x", [1], "m")
    Philote.add_input!(discipline, "y", [1], "m")

    # Declare outputs
    Philote.add_output!(discipline, "f_xy", [1], "m**2")

    # Declare partials (gradients)
    Philote.declare_partials!(discipline, "f_xy", "x")
    Philote.declare_partials!(discipline, "f_xy", "y")

    # Set metadata
    meta = Philote.get_metadata(discipline)
    meta.name = "ParaboloidDiscipline"
    meta.version = "0.1.0"
end

# Compute outputs from inputs
function Philote.compute(discipline::ParaboloidDiscipline,
                        inputs::Dict{String, <:AbstractArray{Float64}})
    x = inputs["x"][1]
    y = inputs["y"][1]

    f_xy = (x - 3.0)^2 + x * y + (y + 4.0)^2 - 3.0
    f_xy = discipline.scale_factor * f_xy + discipline.offset

    return Dict("f_xy" => [f_xy])
end

# Compute analytical gradients
function Philote.compute_partials(discipline::ParaboloidDiscipline,
                                  inputs::Dict{String, <:AbstractArray{Float64}})
    x = inputs["x"][1]
    y = inputs["y"][1]

    df_dx = discipline.scale_factor * (2.0 * (x - 3.0) + y)
    df_dy = discipline.scale_factor * (2.0 * (y + 4.0) + x)

    return Dict(
        "f_xy" => Dict(
            "x" => [df_dx],
            "y" => [df_dy]
        )
    )
end

# Set discipline options from configuration
function Philote.set_options!(discipline::ParaboloidDiscipline,
                              options::Dict{String, <:Any})
    if haskey(options, "scale_factor")
        discipline.scale_factor = Float64(options["scale_factor"])
    end
    if haskey(options, "offset")
        discipline.offset = Float64(options["offset"])
    end
end
```

#### Required Methods for Explicit Disciplines

- `setup!(discipline)` - Declare inputs, outputs, options, and partials
- `compute(discipline, inputs)` - Compute outputs from inputs, returns `Dict{String, Array}`
- `compute_partials(discipline, inputs)` - Compute gradients, returns nested `Dict`
- `set_options!(discipline, options)` - Set custom options (optional but recommended)

### Implicit Disciplines

Implicit disciplines solve residual equations where outputs must satisfy `R(inputs, outputs) = 0` rather than being directly computed.

Here's an example that solves a quadratic equation:

\begin{align}
a x^2 + b x + c &= 0
\end{align}

```julia
using Philote

mutable struct QuadraticDiscipline <: Philote.ImplicitDiscipline
    tolerance::Float64

    function QuadraticDiscipline()
        new(1e-10)
    end
end

function Philote.setup!(discipline::QuadraticDiscipline)
    # Define inputs (coefficients)
    Philote.add_input!(discipline, "a", [1], "unitless")
    Philote.add_input!(discipline, "b", [1], "unitless")
    Philote.add_input!(discipline, "c", [1], "unitless")

    # Define output (solution)
    Philote.add_output!(discipline, "x", [1], "unitless")

    # Define residual (r = a*x^2 + b*x + c)
    Philote.add_residual!(discipline, "r", [1], "unitless")

    # Declare partial derivatives
    Philote.declare_partials!(discipline, "r", "a")
    Philote.declare_partials!(discipline, "r", "b")
    Philote.declare_partials!(discipline, "r", "c")
    Philote.declare_partials!(discipline, "r", "x")
end

# Compute residuals given inputs and outputs
function Philote.compute_residuals(discipline::QuadraticDiscipline,
                                   inputs::Dict{String,Array},
                                   outputs::Dict{String,Array})
    a = inputs["a"][1]
    b = inputs["b"][1]
    c = inputs["c"][1]
    x = outputs["x"][1]

    # Residual: r = a*x^2 + b*x + c
    r = a * x^2 + b * x + c

    return Dict("r" => [r])
end

# Solve for outputs that satisfy residuals
function Philote.solve_residuals(discipline::QuadraticDiscipline,
                                 inputs::Dict{String,Array},
                                 outputs::Dict{String,Array})
    a = inputs["a"][1]
    b = inputs["b"][1]
    c = inputs["c"][1]

    # Solve using quadratic formula
    discriminant = b^2 - 4*a*c

    if discriminant < 0
        error("No real solution: discriminant is negative")
    end

    x = (-b + sqrt(discriminant)) / (2*a)

    # Update output in place
    outputs["x"][1] = x
end

# Compute partial derivatives of residuals
function Philote.residual_partials(discipline::QuadraticDiscipline,
                                   inputs::Dict{String,Array},
                                   outputs::Dict{String,Array})
    a = inputs["a"][1]
    b = inputs["b"][1]
    x = outputs["x"][1]

    # Compute partials of r = a*x^2 + b*x + c
    dr_da = x^2
    dr_db = x
    dr_dc = 1.0
    dr_dx = 2*a*x + b

    return Dict(
        "r" => Dict(
            "a" => reshape([dr_da], 1, 1),
            "b" => reshape([dr_db], 1, 1),
            "c" => reshape([dr_dc], 1, 1),
            "x" => reshape([dr_dx], 1, 1)
        )
    )
end
```

#### Required Methods for Implicit Disciplines

- `setup!(discipline)` - Declare inputs, outputs, residuals, and partials
- `compute_residuals(discipline, inputs, outputs)` - Evaluate residual equations
- `solve_residuals(discipline, inputs, outputs)` - Solve for outputs (modifies `outputs` in-place)
- `residual_partials(discipline, inputs, outputs)` - Compute Jacobian of residuals
- `set_options!(discipline, options)` - Set custom options (optional)

## Configuration Files

Julia disciplines are configured using YAML files that specify the discipline type, file location, and server settings.

### Explicit Discipline Configuration

Here's a configuration for the paraboloid example:

```yaml
# paraboloid.yaml
discipline:
  # Type of discipline: "explicit" or "implicit"
  kind: explicit

  # Path to Julia file (relative to config or absolute)
  julia_file: paraboloid.jl

  # Name of the Julia type/struct to instantiate
  julia_type: ParaboloidDiscipline

  # Optional: discipline-specific options
  options:
    scale_factor: 2.0
    offset: 10.0

server:
  # gRPC server address
  # Use [::]:PORT for all interfaces (IPv4 and IPv6)
  # Use localhost:PORT for localhost only
  address: "[::]:50051"

  # Maximum number of worker threads
  max_workers: 10
```

### Implicit Discipline Configuration

Configuration for the quadratic solver:

```yaml
# quadratic.yaml
discipline:
  kind: implicit
  julia_file: quadratic.jl
  julia_type: QuadraticDiscipline

server:
  address: "[::]:50052"
  max_workers: 10
```

### Configuration Schema

**Discipline Section:**
- `kind` (required): `"explicit"` or `"implicit"`
- `julia_file` (required): Path to `.jl` file containing the discipline
- `julia_type` (required): Name of the Julia struct to instantiate
- `options` (optional): Dictionary of custom options passed to `set_options!()`

**Server Section:**
- `address` (required): gRPC server address (e.g., `"[::]:50051"`)
- `max_workers` (optional): Thread pool size (default: 10)

:::{note}
Paths in `julia_file` can be relative to the configuration file or absolute. The wrapper automatically resolves relative paths.
:::

## Serving Julia Disciplines

Once you have a Julia discipline and configuration file, you can serve it using the `philote-julia-serve` command-line tool.

### Starting a Server

```bash
philote-julia-serve config.yaml
```

This will:
1. Load the Julia discipline from the specified file
2. Instantiate the Julia type
3. Apply any options from the configuration
4. Start a gRPC server at the configured address

Example output:
```
Loading Julia discipline from: /path/to/paraboloid.jl
Instantiating Julia type: ParaboloidDiscipline
Starting gRPC server at [::]:50051
Server started successfully. Press Ctrl+C to stop.
```

### Stopping a Server

Press `Ctrl+C` to gracefully shut down the server:

```
^C
Shutting down server...
Server stopped.
```

## Connecting Clients

Once the Julia discipline server is running, you can connect to it from any Philote client (Python, C++, or other implementations).

### Python Client Example

```python
import grpc
from philote_mdo.general import RemoteExplicitDiscipline
import numpy as np

# Connect to the Julia discipline server
channel = grpc.insecure_channel('localhost:50051')
discipline = RemoteExplicitDiscipline(channel)

# Get discipline metadata
metadata = discipline.get_metadata()
print(f"Connected to: {metadata.name} v{metadata.version}")
print(f"Inputs: {list(metadata.inputs.keys())}")
print(f"Outputs: {list(metadata.outputs.keys())}")

# Evaluate the discipline
inputs = {
    'x': np.array([4.0]),
    'y': np.array([2.0])
}
outputs = discipline.compute(inputs)
print(f"f_xy = {outputs['f_xy'][0]}")

# Compute gradients
partials = discipline.compute_partials(inputs)
print(f"∂f_xy/∂x = {partials['f_xy', 'x'][0]}")
print(f"∂f_xy/∂y = {partials['f_xy', 'y'][0]}")

# Clean up
channel.close()
```

### OpenMDAO Integration

Julia disciplines work seamlessly with OpenMDAO using the `RemoteExplicitComponent` or `RemoteImplicitComponent` wrappers:

```python
import openmdao.api as om
from philote_mdo.openmdao import RemoteExplicitComponent

# Create an OpenMDAO problem
prob = om.Problem()

# Add the Julia discipline as a remote component
prob.model.add_subsystem(
    'paraboloid',
    RemoteExplicitComponent(address='localhost:50051'),
    promotes=['*']
)

# Add optimizer
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

# Add design variables and objective
prob.model.add_design_var('x', lower=-10, upper=10)
prob.model.add_design_var('y', lower=-10, upper=10)
prob.model.add_objective('f_xy')

# Set up and run optimization
prob.setup()
prob.set_val('x', 3.0)
prob.set_val('y', -4.0)

prob.run_driver()

print(f"Optimal x: {prob.get_val('x')[0]:.4f}")
print(f"Optimal y: {prob.get_val('y')[0]:.4f}")
print(f"Minimum f_xy: {prob.get_val('f_xy')[0]:.4f}")
```

For implicit disciplines, use `RemoteImplicitComponent` instead. See {ref}`tutorials:openmdao` for more details on OpenMDAO integration.

## Advanced Topics

### Units Support

Julia disciplines fully support unit specifications. All inputs, outputs, and residuals can have units specified using the same syntax as OpenMDAO (e.g., `"m"`, `"kg"`, `"m/s**2"`).

```julia
Philote.add_input!(discipline, "velocity", [3], "m/s")
Philote.add_output!(discipline, "force", [3], "N")
```

See {ref}`tutorials:units` for more information about unit handling in Philote.

### Shape and Array Variables

Julia disciplines support multidimensional arrays. Specify shapes as tuples when declaring variables:

```julia
# Vector input (3 elements)
Philote.add_input!(discipline, "position", [3], "m")

# Matrix output (3x3)
Philote.add_output!(discipline, "stiffness_matrix", [3, 3], "N/m")
```

Input and output dictionaries contain Julia `Array` objects that can be indexed and manipulated using standard Julia array operations.

### Custom Options

The options system allows you to parameterize your disciplines. Options declared in `setup!()` can be set via the YAML configuration:

```julia
# In Julia discipline
function Philote.setup!(discipline::MyDiscipline)
    Philote.add_option!(discipline, "tolerance", "float")
    Philote.add_option!(discipline, "max_iterations", "int")
    Philote.add_option!(discipline, "method", "string")
    # ...
end

function Philote.set_options!(discipline::MyDiscipline, options::Dict)
    if haskey(options, "tolerance")
        discipline.tolerance = Float64(options["tolerance"])
    end
    if haskey(options, "max_iterations")
        discipline.max_iterations = Int(options["max_iterations"])
    end
    if haskey(options, "method")
        discipline.method = String(options["method"])
    end
end
```

```yaml
# In YAML configuration
discipline:
  # ...
  options:
    tolerance: 1.0e-8
    max_iterations: 100
    method: "newton"
```

### Metadata Discovery

The Julia wrapper automatically discovers the discipline interface by calling `setup!()` and inspecting the metadata. This means clients can query the discipline to learn about its inputs, outputs, and capabilities without prior knowledge.

```julia
# Julia side - metadata is automatically built during setup!()
meta = Philote.get_metadata(discipline)
meta.name = "MyCustomDiscipline"
meta.version = "1.0.0"
meta.description = "A custom Julia discipline"
```

```python
# Python client side - query metadata
metadata = remote_discipline.get_metadata()
print(f"Name: {metadata.name}")
print(f"Inputs: {metadata.inputs}")
print(f"Outputs: {metadata.outputs}")
```

## Troubleshooting

### Common Issues

**Julia file not found:**
```
FileNotFoundError: Julia file not found: paraboloid.jl
```
- Ensure the `julia_file` path is correct (relative to config file or absolute)
- Check that the `.jl` file exists and is readable

**Invalid Julia type:**
```
AttributeError: Julia module has no attribute 'ParaboloidDiscipline'
```
- Verify the `julia_type` name matches the struct name in your `.jl` file
- Ensure the struct is exported or fully qualified

**Port already in use:**
```
RuntimeError: Failed to bind to port 50051
```
- Another process is using the port
- Change the `address` port number in your configuration
- Stop the other server or use `lsof -i :50051` to identify it

**Missing dependencies:**
```
ModuleNotFoundError: No module named 'juliacall'
```
- Install Julia dependencies: `pip install philote-mdo[julia]`

### Debugging Tips

1. **Test your Julia discipline standalone** before serving it:
   ```julia
   include("paraboloid.jl")
   d = ParaboloidDiscipline()
   Philote.setup!(d)
   inputs = Dict("x" => [1.0], "y" => [2.0])
   outputs = Philote.compute(d, inputs)
   println(outputs)
   ```

2. **Check server logs** for detailed error messages when the server fails to start

3. **Verify gRPC connectivity** using a simple client test before integrating with OpenMDAO

4. **Use print statements** in Julia methods during development (they'll appear in server logs)

## Complete Examples

Complete working examples are provided in the Philote-Python repository:

- **Explicit discipline:** `examples/julia/paraboloid.jl`
- **Implicit discipline:** `examples/julia/quadratic.jl`
- **Configurations:** `examples/julia/configs/*.yaml`

These examples demonstrate best practices and can serve as templates for your own Julia disciplines.

## Summary

The Julia integration enables you to:

- Write high-performance disciplines in pure Julia
- Leverage Julia's numerical computing strengths
- Serve disciplines via Python's gRPC infrastructure
- Integrate seamlessly with OpenMDAO and other Philote clients
- Achieve zero-copy data transfer between Python and Julia

Key steps:
1. Write a Julia discipline implementing the Philote.jl interface
2. Create a YAML configuration file
3. Serve the discipline with `philote-julia-serve`
4. Connect from any Philote client

For more information on the Philote.jl API, see the [Philote.jl documentation](https://github.com/MDO-Standards/Philote-Julia).

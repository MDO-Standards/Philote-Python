(tutorials:openmdao)=
# Calling Philote Disciplines from OpenMDAO

This guide explains how to integrate Philote disciplines into OpenMDAO workflows using remote client components. Philote provides OpenMDAO components that act as clients to remote Philote servers, enabling distributed computing and language interoperability.

## Overview

Philote offers two main OpenMDAO component types for integrating remote disciplines:

- **`RemoteExplicitComponent`**: For explicit disciplines where outputs are computed directly from inputs
- **`RemoteImplicitComponent`**: For implicit disciplines that solve residual equations R(inputs, outputs) = 0

Both components automatically discover the server's interface and handle all communication transparently, making remote disciplines appear as native OpenMDAO components.

### Key Benefits

- **Distributed Computing**: Run expensive analyses on remote servers
- **Language Interoperability**: Use disciplines written in any language supported by Philote
- **Seamless Integration**: Remote disciplines work exactly like local OpenMDAO components
- **Automatic Interface Discovery**: No manual specification of inputs/outputs required
- **Derivative Support**: Automatic partial derivative computation for optimization

## Prerequisites

Before using Philote OpenMDAO components, ensure:

1. A Philote server is running and accessible
2. The gRPC channel can connect to the server
3. OpenMDAO and Philote packages are installed

## Remote Explicit Components

### Basic Usage

Explicit components compute outputs directly from inputs. Here's a complete example:

```python
import grpc
import openmdao.api as om
import philote_mdo.openmdao as pmom

# Create gRPC channel to Philote server
channel = grpc.insecure_channel("localhost:50051")

# Create OpenMDAO problem
prob = om.Problem()
model = prob.model

# Add remote explicit component
model.add_subsystem(
    "analysis",
    pmom.RemoteExplicitComponent(channel=channel),
    promotes=["*"]
)

# Setup and run
prob.setup()
prob["x"] = 2.0
prob["y"] = 3.0
prob.run_model()

print(f"Result: {prob['f_xy']}")
```

### Advanced Example with Options

```python
import grpc
import openmdao.api as om
import philote_mdo.openmdao as pmom

# Create channel with server options
channel = grpc.insecure_channel("localhost:50051")

# Create component with server-specific options
component = pmom.RemoteExplicitComponent(
    channel=channel,
    num_par_fd=4,  # OpenMDAO finite difference parallelization
    tolerance=1e-8,  # Server-specific option
    max_iterations=100  # Server-specific option
)

# Create problem and add component
prob = om.Problem()
prob.model.add_subsystem("remote_analysis", component, promotes=["*"])

# Add optimizer for demonstration
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"

# Add design variable and objective
prob.model.add_design_var("x", lower=-10, upper=10)
prob.model.add_objective("f_xy")

# Setup and optimize
prob.setup()
prob["x"] = 1.0
prob["y"] = 1.0

prob.run_driver()
print(f"Optimal x: {prob['x']}")
print(f"Optimal objective: {prob['f_xy']}")
```

## Remote Implicit Components

### Basic Usage

Implicit components solve equations of the form R(inputs, outputs) = 0:

```python
import grpc
import openmdao.api as om
import philote_mdo.openmdao as pmom

# Create gRPC channel
channel = grpc.insecure_channel("localhost:50051")

# Create OpenMDAO problem
prob = om.Problem()
model = prob.model

# Add remote implicit component
model.add_subsystem(
    "implicit_analysis",
    pmom.RemoteImplicitComponent(channel=channel),
    promotes=["*"]
)

# Setup and run
prob.setup()

# Set inputs for quadratic equation: ax^2 + bx + c = 0
prob["a"] = 1.0
prob["b"] = -5.0
prob["c"] = 6.0

prob.run_model()
print(f"Solution: x = {prob['x']}")  # Should find x = 2 or x = 3
```

### Implicit Component with Nonlinear Solver

For more complex implicit systems, you may need to configure OpenMDAO solvers:

```python
import grpc
import openmdao.api as om
import philote_mdo.openmdao as pmom

# Create problem with nonlinear solver
prob = om.Problem()
model = prob.model

# Add implicit component
model.add_subsystem(
    "implicit_system",
    pmom.RemoteImplicitComponent(channel=grpc.insecure_channel("localhost:50051")),
    promotes=["*"]
)

# Configure nonlinear solver for implicit system
model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
model.nonlinear_solver.options["iprint"] = 2
model.nonlinear_solver.options["maxiter"] = 20
model.linear_solver = om.DirectSolver()

# Setup and run
prob.setup()
prob["input_param"] = 5.0
prob.run_model()

print(f"Converged solution: {prob['output_var']}")
```

## Working with Multiple Remote Components

You can combine multiple remote components in a single OpenMDAO model:

```python
import grpc
import openmdao.api as om
import philote_mdo.openmdao as pmom

# Create channels to different servers
channel1 = grpc.insecure_channel("server1:50051")
channel2 = grpc.insecure_channel("server2:50052")

# Create problem with multiple remote components
prob = om.Problem()
model = prob.model

# Add multiple remote components
model.add_subsystem(
    "aerodynamics",
    pmom.RemoteExplicitComponent(channel=channel1),
    promotes_inputs=["mach", "alpha", "altitude"],
    promotes_outputs=["CL", "CD"]
)

model.add_subsystem(
    "structures",
    pmom.RemoteImplicitComponent(channel=channel2),
    promotes_inputs=["loads", "material_props"],
    promotes_outputs=["stress", "displacement"]
)

# Connect components
model.connect("CL", "structures.loads")

# Setup and run coupled analysis
prob.setup()
prob["mach"] = 0.8
prob["alpha"] = 2.0
prob["altitude"] = 35000
prob["material_props"] = [...]

prob.run_model()
```

## Error Handling and Debugging

### Connection Issues

```python
import grpc
import philote_mdo.openmdao as pmom

try:
    channel = grpc.insecure_channel("unreachable:50051")
    component = pmom.RemoteExplicitComponent(channel=channel)
except grpc.RpcError as e:
    print(f"Failed to connect to server: {e}")
    print("Check that the server is running and accessible")
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Server Debugging

Enable OpenMDAO's debug output to see server communication:

```python
import openmdao.api as om

# Enable debug output
prob = om.Problem()
prob.model.add_subsystem("remote", component)

# Setup with debug info
prob.setup()
prob.set_solver_print(level=2)  # Print solver information
prob.run_model()
```

## Performance Considerations

### Parallel Finite Differences

For components that don't provide analytic derivatives:

```python
component = pmom.RemoteExplicitComponent(
    channel=channel,
    num_par_fd=8  # Use 8 parallel processes for finite differences
)
```

### Connection Pooling

For multiple components connecting to the same server:

```python
# Reuse channels when possible
shared_channel = grpc.insecure_channel("localhost:50051")

comp1 = pmom.RemoteExplicitComponent(channel=shared_channel)
comp2 = pmom.RemoteExplicitComponent(channel=shared_channel)
```

### Server-Side Optimization

- Ensure servers are running on appropriate hardware
- Use analytic derivatives when possible (server-side implementation)
- Consider caching for expensive computations (server-side)

## Best Practices

1. **Error Handling**: Always wrap component creation in try-catch blocks
2. **Connection Management**: Reuse gRPC channels when connecting to the same server
3. **Option Validation**: Server options are discovered automatically but validate important values
4. **Solver Configuration**: Configure appropriate solvers for implicit components
5. **Performance**: Use analytic derivatives when available from the server
6. **Debugging**: Enable OpenMDAO debug output to troubleshoot communication issues

## Troubleshooting

### Common Issues

1. **"No channel provided" Error**
   ```python
   # Wrong: Missing channel
   component = pmom.RemoteExplicitComponent()
   
   # Correct: Provide valid channel
   channel = grpc.insecure_channel("localhost:50051")
   component = pmom.RemoteExplicitComponent(channel=channel)
   ```

2. **Server Connection Failures**
   - Verify server is running: `telnet localhost 50051`
   - Check firewall settings
   - Verify correct hostname/port

3. **Variable Name Mismatches**
   - Use `prob.model.list_inputs()` and `prob.model.list_outputs()` after setup
   - Server variables are automatically discovered and named

4. **Convergence Issues with Implicit Components**
   - Check residual values: use OpenMDAO's debug output
   - Adjust solver tolerances and iteration limits
   - Verify initial guesses are reasonable

### Debug Tools

```python
# After problem setup, inspect the remote component
prob.setup()

# List all inputs and outputs
prob.model.list_inputs(print_arrays=True)
prob.model.list_outputs(print_arrays=True)

# Check partial derivatives
prob.check_partials(compact_print=True)
```

## API Reference

### RemoteExplicitComponent

```{eval-rst}
.. autoclass:: philote_mdo.openmdao.explicit.RemoteExplicitComponent
   :members:
   :undoc-members:
   :show-inheritance:
```

### RemoteImplicitComponent

```{eval-rst}
.. autoclass:: philote_mdo.openmdao.implicit.RemoteImplicitComponent
   :members:
   :undoc-members:
   :show-inheritance:
```
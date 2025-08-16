
# Creating Implicit Disciplines

This guide explains how to create, serve, and use implicit disciplines in Philote. Implicit disciplines solve equations of the form R(inputs, outputs) = 0, where outputs are implicitly defined by the inputs through residual equations. Unlike explicit disciplines that compute outputs directly, implicit disciplines require solving nonlinear equations.

## Overview

Implicit disciplines are essential for modeling systems where:
- Outputs cannot be computed directly from inputs
- Equilibrium conditions must be satisfied
- Coupled physics require simultaneous solution
- Conservation laws or constraints must be enforced

### Key Concepts

**Residual Equations**: Mathematical expressions R(inputs, outputs) that must equal zero at the solution.

**Implicit Solving**: Finding output values that satisfy R(inputs, outputs) = 0 for given inputs.

**Jacobian Matrix**: Partial derivatives dR/d[inputs,outputs] used for optimization and sensitivity analysis.

### Mathematical Formulation

For implicit disciplines, we solve:
```
R(x, y) = 0
```
Where:
- `x` are the inputs (design variables, parameters)  
- `y` are the outputs (state variables, unknowns)
- `R` is the residual function

The goal is to find `y` such that `R(x, y) = 0` for given inputs `x`.

## Creating Implicit Disciplines

### Basic Structure

All implicit disciplines inherit from `ImplicitDiscipline` and must implement four key methods:

```python
import numpy as np
import philote_mdo.general as pmdo

class MyImplicitDiscipline(pmdo.ImplicitDiscipline):
    def setup(self):
        # Define inputs and outputs
        pass
    
    def setup_partials(self):
        # Declare which partial derivatives will be computed
        pass
        
    def compute_residuals(self, inputs, outputs, residuals):
        # Compute R(inputs, outputs)
        pass
        
    def solve_residuals(self, inputs, outputs):
        # Solve R(inputs, outputs) = 0 for outputs
        pass
        
    def residual_partials(self, inputs, outputs, partials):
        # Compute dR/d[inputs,outputs]
        pass
```

### Example 1: Quadratic Equation Solver

Let's implement a solver for the quadratic equation ax² + bx + c = 0:

```python
import numpy as np
import philote_mdo.general as pmdo

class QuadraticSolver(pmdo.ImplicitDiscipline):
    def setup(self):
        # Define inputs: coefficients of quadratic equation
        self.add_input('a', shape=(1,))
        self.add_input('b', shape=(1,))
        self.add_input('c', shape=(1,))
        
        # Define output: solution to the equation
        self.add_output('x', shape=(1,))
    
    def setup_partials(self):
        # Declare all partial derivatives that will be computed
        self.declare_partials('x', 'a')  # dR/da
        self.declare_partials('x', 'b')  # dR/db  
        self.declare_partials('x', 'c')  # dR/dc
        self.declare_partials('x', 'x')  # dR/dx
    
    def compute_residuals(self, inputs, outputs, residuals):
        """Compute residual: R = ax² + bx + c"""
        a = inputs['a']
        b = inputs['b'] 
        c = inputs['c']
        x = outputs['x']
        
        residuals['x'] = a * x**2 + b * x + c
    
    def solve_residuals(self, inputs, outputs):
        """Solve quadratic equation analytically"""
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        
        # Use quadratic formula: x = (-b + sqrt(b²-4ac)) / 2a
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            raise ValueError("No real solution exists")
            
        outputs['x'] = (-b + np.sqrt(discriminant)) / (2*a)
    
    def residual_partials(self, inputs, outputs, partials):
        """Compute Jacobian of residual equation"""
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        
        # Partial derivatives of R = ax² + bx + c
        partials['x', 'a'] = x**2     # dR/da
        partials['x', 'b'] = x        # dR/db
        partials['x', 'c'] = 1.0      # dR/dc
        partials['x', 'x'] = 2*a*x + b # dR/dx
```

### Example 2: Coupled System of Equations

For more complex systems with multiple residuals:

```python
class CoupledSystem(pmdo.ImplicitDiscipline):
    def setup(self):
        # Inputs: parameters
        self.add_input('p1', shape=(1,))
        self.add_input('p2', shape=(1,))
        
        # Outputs: state variables
        self.add_output('x1', shape=(1,))
        self.add_output('x2', shape=(1,))
    
    def setup_partials(self):
        # Declare partials for all residual-variable pairs
        self.declare_partials('x1', ['p1', 'x1', 'x2'])
        self.declare_partials('x2', ['p2', 'x1', 'x2'])
    
    def compute_residuals(self, inputs, outputs, residuals):
        """
        Solve coupled system:
        R1 = x1² + x2 - p1 = 0
        R2 = x1 + x2² - p2 = 0  
        """
        p1, p2 = inputs['p1'], inputs['p2']
        x1, x2 = outputs['x1'], outputs['x2']
        
        residuals['x1'] = x1**2 + x2 - p1
        residuals['x2'] = x1 + x2**2 - p2
    
    def solve_residuals(self, inputs, outputs):
        """Solve using Newton's method"""
        p1, p2 = inputs['p1'], inputs['p2']
        x1, x2 = outputs['x1'], outputs['x2']  # Initial guess
        
        max_iter = 20
        tolerance = 1e-8
        
        for i in range(max_iter):
            # Compute residuals
            residuals = {}
            self.compute_residuals(inputs, outputs, residuals)
            
            # Check convergence
            r1, r2 = residuals['x1'], residuals['x2']
            if np.abs(r1) < tolerance and np.abs(r2) < tolerance:
                break
            
            # Compute Jacobian
            partials = {}
            self.residual_partials(inputs, outputs, partials)
            
            # Newton update: [x1; x2] -= J^(-1) * [r1; r2]
            J11 = partials['x1', 'x1']  # dR1/dx1
            J12 = partials['x1', 'x2']  # dR1/dx2  
            J21 = partials['x2', 'x1']  # dR2/dx1
            J22 = partials['x2', 'x2']  # dR2/dx2
            
            det = J11*J22 - J12*J21
            dx1 = -(J22*r1 - J12*r2) / det
            dx2 = -(-J21*r1 + J11*r2) / det
            
            outputs['x1'] += dx1
            outputs['x2'] += dx2
        else:
            raise RuntimeError("Newton method failed to converge")
    
    def residual_partials(self, inputs, outputs, partials):
        """Compute Jacobian matrix"""
        x1, x2 = outputs['x1'], outputs['x2']
        
        # Partials of R1 = x1² + x2 - p1
        partials['x1', 'p1'] = -1.0
        partials['x1', 'x1'] = 2*x1
        partials['x1', 'x2'] = 1.0
        
        # Partials of R2 = x1 + x2² - p2  
        partials['x2', 'p2'] = -1.0
        partials['x2', 'x1'] = 1.0
        partials['x2', 'x2'] = 2*x2
```

## Serving Implicit Disciplines

### Creating a Server

To serve an implicit discipline over gRPC:

```python
from concurrent import futures
import grpc
import philote_mdo.general as pmdo

def run_server():
    # Create the discipline
    discipline = QuadraticSolver()
    
    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Create and attach implicit server
    impl_server = pmdo.ImplicitServer(discipline=discipline)
    impl_server.attach_to_server(server)
    
    # Start server
    server.add_insecure_port('[::]:50051')
    server.start()
    print('Implicit discipline server started on port 50051')
    server.wait_for_termination()

if __name__ == '__main__':
    run_server()
```

### Server Configuration

For production servers, consider:

```python
def run_production_server():
    discipline = QuadraticSolver()
    
    # Configure server with more workers for concurrent clients
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=50),
        options=[
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 300000)
        ]
    )
    
    impl_server = pmdo.ImplicitServer(discipline=discipline)
    impl_server.attach_to_server(server)
    
    # Use secure connection in production
    private_key = open('server.key', 'rb').read()
    certificate_chain = open('server.crt', 'rb').read()
    credentials = grpc.ssl_server_credentials([(private_key, certificate_chain)])
    
    server.add_secure_port('[::]:443', credentials)
    server.start()
    print('Secure implicit discipline server started on port 443')
    server.wait_for_termination()
```

## Using Implicit Disciplines (Client Side)

### Basic Client Usage

```python
import grpc
import numpy as np
import philote_mdo.general as pmdo

# Connect to server
channel = grpc.insecure_channel('localhost:50051')
client = pmdo.ImplicitClient(channel)

# Define inputs for quadratic equation: x² - 5x + 6 = 0
inputs = {
    'a': np.array([1.0]),
    'b': np.array([-5.0]), 
    'c': np.array([6.0])
}

# Solve the implicit equations
solution = client.run_solve_residuals(inputs)
print(f"Solution: x = {solution['x'][0]}")  # Should be 2.0 or 3.0

# Verify solution by computing residuals
outputs = {'x': solution['x']}
residuals = client.run_compute_residuals(inputs, outputs)
print(f"Residual: {residuals['x'][0]}")  # Should be close to 0.0

# Compute Jacobian for sensitivity analysis
jacobian = client.run_residual_gradients(inputs, outputs)
print(f"dR/da = {jacobian[('x', 'a')][0]}")  # x² = 4.0
print(f"dR/db = {jacobian[('x', 'b')][0]}")  # x = 2.0  
print(f"dR/dc = {jacobian[('x', 'c')][0]}")  # 1.0
print(f"dR/dx = {jacobian[('x', 'x')][0]}")  # 2ax + b = -1.0
```

### Advanced Client Operations

```python
def analyze_quadratic_sensitivity():
    """Analyze how solution changes with coefficients"""
    channel = grpc.insecure_channel('localhost:50051')
    client = pmdo.ImplicitClient(channel)
    
    # Base case
    base_inputs = {'a': np.array([1.0]), 'b': np.array([-5.0]), 'c': np.array([6.0])}
    base_solution = client.run_solve_residuals(base_inputs)
    base_x = base_solution['x'][0]
    
    print(f"Base solution: x = {base_x}")
    
    # Sensitivity to coefficient 'a'
    perturbed_inputs = base_inputs.copy()
    perturbed_inputs['a'] = np.array([1.1])  # 10% increase
    
    perturbed_solution = client.run_solve_residuals(perturbed_inputs)
    perturbed_x = perturbed_solution['x'][0]
    
    sensitivity = (perturbed_x - base_x) / (0.1 * base_inputs['a'][0])
    print(f"Finite difference sensitivity dx/da ≈ {sensitivity}")
    
    # Compare with analytical sensitivity from Jacobian
    outputs = {'x': base_solution['x']}
    jacobian = client.run_residual_gradients(base_inputs, outputs)
    
    # For implicit function theorem: dx/da = -(dR/da) / (dR/dx)
    dR_da = jacobian[('x', 'a')][0]
    dR_dx = jacobian[('x', 'x')][0]
    analytical_sensitivity = -dR_da / dR_dx
    
    print(f"Analytical sensitivity dx/da = {analytical_sensitivity}")

def solve_parameter_sweep():
    """Solve for multiple parameter values"""
    channel = grpc.insecure_channel('localhost:50051')
    client = pmdo.ImplicitClient(channel)
    
    # Sweep parameter 'c' while keeping 'a' and 'b' fixed
    c_values = np.linspace(0, 10, 21)
    solutions = []
    
    for c in c_values:
        inputs = {'a': np.array([1.0]), 'b': np.array([-5.0]), 'c': np.array([c])}
        
        try:
            solution = client.run_solve_residuals(inputs)
            x = solution['x'][0]
            solutions.append((c, x))
            print(f"c = {c:4.1f}, x = {x:6.3f}")
        except grpc.RpcError as e:
            print(f"c = {c:4.1f}, Failed to solve: {e}")
    
    return solutions
```

### Error Handling

```python
def robust_client_usage():
    """Demonstrate proper error handling"""
    try:
        channel = grpc.insecure_channel('localhost:50051')
        client = pmdo.ImplicitClient(channel)
        
        # This should fail - no real solution
        inputs = {'a': np.array([1.0]), 'b': np.array([1.0]), 'c': np.array([1.0])}
        
        solution = client.run_solve_residuals(inputs)
        print(f"Unexpected solution: {solution}")
        
    except grpc.RpcError as e:
        print(f"Server error: {e.code()}")
        print(f"Details: {e.details()}")
        
    except Exception as e:
        print(f"Other error: {e}")
        
    finally:
        # Clean up connection
        if 'channel' in locals():
            channel.close()
```


## API Reference

### ImplicitDiscipline

```{eval-rst}
.. autoclass:: philote_mdo.general.implicit_discipline.ImplicitDiscipline
   :members:
   :undoc-members:
   :show-inheritance:
```

### ImplicitServer

```{eval-rst}
.. autoclass:: philote_mdo.general.implicit_server.ImplicitServer
   :members:
   :undoc-members:
   :show-inheritance:
```

### ImplicitClient

```{eval-rst}
.. autoclass:: philote_mdo.general.implicit_client.ImplicitClient
   :members:
   :undoc-members:
   :show-inheritance:
```

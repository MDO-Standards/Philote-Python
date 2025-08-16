# OpenMDAO Group Wrapping Discipline

The `OpenMdaoSubProblem` class allows you to wrap existing OpenMDAO groups as Philote disciplines, enabling integration of OpenMDAO models into Philote workflows. This wrapper creates a bridge between OpenMDAO's group-based architecture and Philote's discipline-based framework.

## Overview

The `OpenMdaoSubProblem` class (`philote_mdo.openmdao.group.OpenMdaoSubProblem`) is a Philote explicit discipline that internally contains and executes an OpenMDAO group. While the Philote discipline interface is explicit, the underlying OpenMDAO group may contain cycles that require nonlinear solvers.

### Key Features

- Wraps any OpenMDAO group as a Philote discipline
- Maps variables between Philote and OpenMDAO namespaces
- Supports automatic partial derivative computation using OpenMDAO's `compute_totals`
- Handles units and multi-dimensional arrays
- Preserves OpenMDAO solver capabilities for cyclic groups

## Basic Usage

### 1. Create and Configure the Wrapper

```python
import openmdao.api as om
from philote_mdo.openmdao.group import OpenMdaoSubProblem

# Create the wrapper discipline
subprob = OpenMdaoSubProblem()

# Add your OpenMDAO group
my_group = MyOpenMDAOGroup()
subprob.add_group(my_group)
```

### 2. Map Variables

Map inputs and outputs between the Philote discipline and the OpenMDAO group:

```python
# Map inputs: (philote_name, openmdao_name, shape, units)
subprob.add_mapped_input('x_local', 'x', shape=(1,), units='m')
subprob.add_mapped_input('design_vars', 'z', shape=(2,), units='')

# Map outputs: (philote_name, openmdao_name, shape, units)
subprob.add_mapped_output('objective', 'obj', shape=(1,), units='')
subprob.add_mapped_output('constraint1', 'con1', shape=(1,), units='')
```

### 3. Declare Partial Derivatives

If you need partial derivatives, declare them for each output-input pair:

```python
# Declare partials: (output_name, input_name)
subprob.declare_subproblem_partial('objective', 'x_local')
subprob.declare_subproblem_partial('objective', 'design_vars')
subprob.declare_subproblem_partial('constraint1', 'x_local')
```

## Complete Example: Simple Linear Function

```python
import numpy as np
import openmdao.api as om
from philote_mdo.openmdao.group import OpenMdaoSubProblem

# Define a simple OpenMDAO group
class SimpleGroup(om.Group):
    def setup(self):
        self.add_subsystem('comp', om.ExecComp('y = 2*x + 1'), promotes=['*'])

# Create and configure the wrapper
subprob = OpenMdaoSubProblem()
subprob.add_group(SimpleGroup())

# Map variables
subprob.add_mapped_input('input_x', 'x')
subprob.add_mapped_output('output_y', 'y')

# Declare partials
subprob.declare_subproblem_partial('output_y', 'input_x')

# Setup the discipline
subprob.setup()

# Use the discipline
inputs = {'input_x': np.array([3.0])}
outputs = {'output_y': np.array([0.0])}

subprob.compute(inputs, outputs)
print(f"Result: {outputs['output_y'][0]}")  # Should print 7.0
```

## Advanced Example: Sellar Problem

The Sellar MDA problem demonstrates wrapping a more complex OpenMDAO group with cycles:

```python
import numpy as np
import openmdao.api as om
from openmdao.test_suite.components.sellar import SellarDis1, SellarDis2
from philote_mdo.openmdao.group import OpenMdaoSubProblem

class SellarMDA(om.Group):
    def setup(self):
        # Create a cycle group
        cycle = self.add_subsystem("cycle", om.Group(), promotes=["*"])
        cycle.add_subsystem("d1", SellarDis1(), 
                           promotes_inputs=["x", "z", "y2"],
                           promotes_outputs=["y1"])
        cycle.add_subsystem("d2", SellarDis2(), 
                           promotes_inputs=["z", "y1"], 
                           promotes_outputs=["y2"])
        
        # Set default values
        cycle.set_input_defaults("x", 1.0)
        cycle.set_input_defaults("z", np.array([5.0, 2.0]))
        
        # Add solvers for the cycle
        cycle.nonlinear_solver = om.NonlinearBlockGS(iprint=0)
        cycle.linear_solver = om.LinearBlockGS(iprint=0)
        
        # Add objective and constraints
        self.add_subsystem("obj_cmp",
                          om.ExecComp("obj = x**2 + z[1] + y1 + exp(-y2)", 
                                     z=np.array([0.0, 0.0]), x=0.0),
                          promotes=["x", "z", "y1", "y2", "obj"])
        
        self.add_subsystem("con_cmp1", 
                          om.ExecComp("con1 = 3.16 - y1"), 
                          promotes=["con1", "y1"])
        self.add_subsystem("con_cmp2", 
                          om.ExecComp("con2 = y2 - 24.0"), 
                          promotes=["con2", "y2"])

# Create wrapper using inheritance
class SellarGroup(OpenMdaoSubProblem):
    def initialize(self):
        self.add_group(SellarMDA())
        
        # Map all variables
        self.add_mapped_input('x', 'x')
        self.add_mapped_input('z', 'z', shape=(2,))
        self.add_mapped_output('obj', 'obj')
        self.add_mapped_output('con1', 'con1')
        self.add_mapped_output('con2', 'con2')
        
        # Declare partials for optimization
        self.declare_subproblem_partial('obj', 'x')
        self.declare_subproblem_partial('obj', 'z')
        self.declare_subproblem_partial('con1', 'x')
        self.declare_subproblem_partial('con1', 'z')
        self.declare_subproblem_partial('con2', 'x')
        self.declare_subproblem_partial('con2', 'z')
```

## API Reference

```{eval-rst}
.. autoclass:: philote_mdo.openmdao.group.OpenMdaoSubProblem
   :members:
   :undoc-members:
   :show-inheritance:
```

## Best Practices

1. **Variable Mapping**: Always map variables before calling `setup()`
2. **Partial Derivatives**: Only declare partials you actually need to avoid unnecessary computation
3. **Units**: Specify units when mapping variables to ensure proper unit conversion
4. **Shapes**: Correctly specify array shapes for multi-dimensional variables
5. **Inheritance**: Consider inheriting from `OpenMdaoSubProblem` for reusable wrappers
6. **Solver Configuration**: Configure OpenMDAO solvers within your group's `setup()` method

## Troubleshooting

### Common Issues

1. **Missing Variables**: Ensure all mapped variables exist in the OpenMDAO group
2. **Shape Mismatches**: Verify that specified shapes match the actual variable shapes
3. **Unit Inconsistencies**: Check that units are compatible between mapped variables
4. **Solver Convergence**: For cyclic groups, ensure appropriate solvers are configured

### Debugging Tips

- Use OpenMDAO's `list_inputs()` and `list_outputs()` to verify available variables
- Check solver convergence with `iprint` settings
- Validate partial derivatives using OpenMDAO's `check_partials()` method
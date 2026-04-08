---
sidebar_position: 1
title: "Explicit Disciplines"
---

# Creating Explicit Disciplines

The [Quick Start](../getting-started/quickstart.md) guide introduces explicit disciplines without elaborating on how they work or how to create them. In this section, we will cover the basics of creating explicit disciplines.

Philote-Python implements disciplines in a similar way to OpenMDAO. We create a class, inheriting from a base class and then specialize some methods to run the calculations we want.

To illustrate this, let us take a look at a simple paraboloid problem (the same problem as in the OpenMDAO documentation, and the same one used in the quick start guide):

$$
f(x,y) = (x-3)^2 + x y + (y+4)^2 - 3
$$

To create a discipline that executes this equation, we create a class and inherit from the `ExplicitDiscipline` class Philote-Python provides. Member functions of the inherited class are overloaded to implement the desired functionality. The most common functions that will need to be overloaded are `setup`, `setup_partials`, `compute`, and in the case of a discipline that offers derivatives, `compute_partials`.

## Setup Functions

First, let us look at the setup member function:

```python
import philote_mdo.general as pmdo

class Paraboloid(pmdo.ExplicitDiscipline):

    def setup(self):
        self.add_input("x", shape=(1,), units="m")
        self.add_input("y", shape=(1,), units="m")

        self.add_output("f_xy", shape=(1,), units="m**2")
```

The `setup` function exists to define the inputs and outputs of a discipline (this purpose is borrowed from the OpenMDAO workflow, which operates the same way). The `add_input` and `add_output` member functions are provided by `ExplicitDiscipline` to define both inputs and outputs. Here, two inputs (`x` and `y`) are defined, each with the shape of 1 (scalar) and with a unit of meters (no physical meaning, purely for demonstration). Furthermore, a scalar output (`f_xy`) was defined with the units of square meters.

While the discipline gradients (or partials) can be defined within the `setup` function, it usually is good practice to separate these definitions into the `setup_partials` function. Behind the scenes, both functions (`setup` and `setup_partials`) are called back-to-back, so there is no actual difference where the gradients are defined. However, using both functions to define the variables and partials may have organizational benefits.

To define the gradient of an output with respect to an input, the `declare_partials` function is invoked:

```python
    def setup_partials(self):
        self.declare_partials("f_xy", "x")
        self.declare_partials("f_xy", "y")
```

## Compute Function

To implement the paraboloid function, the `compute` member function must be defined:

```python
    def compute(self, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]

        outputs["f_xy"] = (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0
```

The `inputs` and `outputs` variables for this function are later passed in by the server as dictionaries with the variable names as the keys.

## Gradient Function

To implement the paraboloid function derivatives, the `compute_partials` member function must be defined:

```python
    def compute_partials(self, inputs, partials):
        x = inputs["x"]
        y = inputs["y"]

        partials["f_xy", "x"] = 2.0 * x - 6.0 + y
        partials["f_xy", "y"] = 2.0 * y + 8.0 + x
```

The `inputs` and `partials` variables for this function are later passed in by the server as dictionaries with the variable names as the keys.

## Summary

In this section we covered the basics of creating an explicit discipline. The entire code for the paraboloid discipline is listed here for completeness:

```python
import philote_mdo.general as pmdo

class Paraboloid(pmdo.ExplicitDiscipline):
    """
    Basic two-dimensional paraboloid example (explicit) discipline.
    """

    def setup(self):
        self.add_input("x", shape=(1,), units="m")
        self.add_input("y", shape=(1,), units="m")

        self.add_output("f_xy", shape=(1,), units="m**2")

    def setup_partials(self):
        self.declare_partials("f_xy", "x")
        self.declare_partials("f_xy", "y")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]

        outputs["f_xy"] = (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0

    def compute_partials(self, inputs, partials):
        x = inputs["x"]
        y = inputs["y"]

        partials["f_xy", "x"] = 2.0 * x - 6.0 + y
        partials["f_xy", "y"] = 2.0 * y + 8.0 + x
```

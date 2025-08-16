# Philote-Python
#
# Copyright 2022-2025 Christopher A. Lupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# This work has been cleared for public release, distribution unlimited, case
# number: AFRL-2023-5713.
#
# The views expressed are those of the authors and do not reflect the
# official guidance or position of the United States Government, the
# Department of Defense or of the United States Air Force.
#
# Statement from DoD: The Appearance of external hyperlinks does not
# constitute endorsement by the United States Department of Defense (DoD) of
# the linked websites, of the information, products, or services contained
# therein. The DoD does not exercise any editorial, security, or other
# control over the information you may find at these locations.
import philote_mdo.general as pmdo


class ImplicitDiscipline(pmdo.Discipline):
    """
    Base class for implementing implicit disciplines in Philote.

    Implicit disciplines solve equations of the form R(inputs, outputs) = 0, where the outputs
    are implicitly defined by the inputs through residual equations. Unlike explicit disciplines
    that compute outputs directly, implicit disciplines require solving nonlinear equations
    to find outputs that satisfy the residual constraints.

    This class provides the framework for implementing implicit disciplines that can be served
    over gRPC and integrated with optimization frameworks like OpenMDAO. Subclasses must
    implement the required methods to define the residual equations, solving strategy, and
    derivative computation.

    Key Features:
        - Residual equation definition via compute_residuals
        - Implicit solving capability via solve_residuals  
        - Jacobian computation for optimization via residual_partials
        - Linear operator support for matrix-free methods via apply_linear
        - Automatic gRPC service generation
        - Integration with optimization frameworks

    Mathematical Formulation:
        For implicit disciplines, we solve: R(x, y) = 0
        Where:
        - x are the inputs (design variables, parameters)
        - y are the outputs (state variables, unknowns)
        - R is the residual function

        The goal is to find y such that R(x, y) = 0 for given inputs x.

    Typical Usage:
        >>> import numpy as np
        >>> import philote_mdo.general as pmdo
        >>>
        >>> class QuadraticSolver(pmdo.ImplicitDiscipline):
        ...     def setup(self):
        ...         self.add_input('a', shape=(1,))
        ...         self.add_input('b', shape=(1,))
        ...         self.add_input('c', shape=(1,))
        ...         self.add_output('x', shape=(1,))
        ...
        ...     def compute_residuals(self, inputs, outputs, residuals):
        ...         a, b, c = inputs['a'], inputs['b'], inputs['c']
        ...         x = outputs['x']
        ...         residuals['x'] = a * x**2 + b * x + c
        ...
        ...     def solve_residuals(self, inputs, outputs):
        ...         a, b, c = inputs['a'], inputs['b'], inputs['c']
        ...         discriminant = b**2 - 4*a*c
        ...         outputs['x'] = (-b + np.sqrt(discriminant)) / (2*a)

    Attributes:
        _is_implicit (bool): Always True for implicit disciplines

    Notes:
        - All methods except __init__ must be implemented by subclasses
        - The discipline automatically supports both local and remote execution
        - Residual equations should be formulated such that R = 0 at the solution
        - Both forward and reverse mode automatic differentiation are supported
    """

    def __init__(self):
        """
        Initialize the implicit discipline.

        Sets up the base discipline and marks this as an implicit discipline type.
        Subclasses should call super().__init__() and then implement the required methods.

        Examples
        --------
        >>> class MyImplicitDiscipline(ImplicitDiscipline):
        ...     def __init__(self):
        ...         super().__init__()
        ...         # Additional initialization if needed
        """
        super().__init__()
        self._is_implicit = True

    def compute_residuals(self, inputs, outputs, residuals):
        """
        Compute residual equations R(inputs, outputs) that must equal zero.

        This method evaluates the residual functions given the current values of inputs
        and outputs. The residuals represent the constraint equations that must be
        satisfied: R(inputs, outputs) = 0. The goal of the implicit solver is to find
        output values that drive these residuals to zero.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values with variable names as keys and numpy arrays as values
        outputs : dict
            Dictionary of current output values (current guess) with variable names as keys
        residuals : dict
            Dictionary to store computed residual values with variable names as keys.
            Must be populated by this method.

        Examples
        --------
        For a quadratic equation ax² + bx + c = 0:

        >>> def compute_residuals(self, inputs, outputs, residuals):
        ...     a = inputs['a']
        ...     b = inputs['b'] 
        ...     c = inputs['c']
        ...     x = outputs['x']
        ...     residuals['x'] = a * x**2 + b * x + c

        For a system of coupled equations:

        >>> def compute_residuals(self, inputs, outputs, residuals):
        ...     x1, x2 = outputs['x1'], outputs['x2']
        ...     p1, p2 = inputs['p1'], inputs['p2']
        ...     residuals['x1'] = x1**2 + x2 - p1
        ...     residuals['x2'] = x1 + x2**2 - p2

        Notes
        -----
        - Residuals should be zero at the solution
        - All output variables should have corresponding residual equations
        - Residual arrays must have the same shape as their corresponding outputs
        - This method is called during both residual evaluation and solving
        """
        raise NotImplementedError("compute_residuals not implemented")

    def solve_residuals(self, inputs, outputs):
        """
        Solve the implicit equations to find outputs that satisfy R(inputs, outputs) = 0.

        This method implements the nonlinear solving strategy to find output values that
        drive the residuals to zero. The specific solving approach (Newton's method,
        fixed-point iteration, analytical solution, etc.) depends on the problem
        characteristics and implementation choice.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values with variable names as keys and numpy arrays as values
        outputs : dict
            Dictionary of output values to be updated with the solution. Modified in-place.

        Examples
        --------
        Analytical solution for quadratic equation:

        >>> def solve_residuals(self, inputs, outputs):
        ...     a, b, c = inputs['a'], inputs['b'], inputs['c']
        ...     discriminant = b**2 - 4*a*c
        ...     outputs['x'] = (-b + np.sqrt(discriminant)) / (2*a)

        Iterative Newton's method:

        >>> def solve_residuals(self, inputs, outputs):
        ...     max_iter = 20
        ...     tolerance = 1e-8
        ...     
        ...     for i in range(max_iter):
        ...         residuals = {}
        ...         self.compute_residuals(inputs, outputs, residuals)
        ...         
        ...         if np.abs(residuals['x']) < tolerance:
        ...             break
        ...             
        ...         # Newton update: x_new = x_old - R/dR_dx
        ...         partials = {}
        ...         self.residual_partials(inputs, outputs, partials)
        ...         outputs['x'] -= residuals['x'] / partials['x', 'x']

        Fixed-point iteration:

        >>> def solve_residuals(self, inputs, outputs):
        ...     # For equation x = f(x), iterate x_{n+1} = f(x_n)
        ...     for i in range(100):
        ...         x_old = outputs['x'].copy()
        ...         outputs['x'] = self._fixed_point_function(inputs, outputs['x'])
        ...         if np.abs(outputs['x'] - x_old) < 1e-8:
        ...             break

        Notes
        -----
        - The outputs dictionary is modified in-place
        - Convergence criteria and tolerances should be appropriate for the problem
        - Consider numerical stability and robustness for different input ranges
        - May throw exceptions for non-convergent cases or ill-conditioned problems
        - Initial guesses in outputs may affect convergence
        """
        raise NotImplementedError("solve_residuals not implemented")

    def residual_partials(self, inputs, outputs, partials):
        """
        Compute partial derivatives of residuals with respect to inputs and outputs.

        This method computes the Jacobian matrix elements dR/dx for optimization and
        sensitivity analysis. For implicit disciplines, this includes both dR/dinputs
        and dR/doutputs terms, which are essential for gradient-based optimization
        and coupled system analysis.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values with variable names as keys
        outputs : dict
            Dictionary of output values with variable names as keys
        partials : dict
            Dictionary to store partial derivatives with (residual, variable) tuples as keys.
            Must be populated by this method with numpy arrays as values.

        Examples
        --------
        For quadratic equation R = ax² + bx + c:

        >>> def residual_partials(self, inputs, outputs, partials):
        ...     a, b, c = inputs['a'], inputs['b'], inputs['c']
        ...     x = outputs['x']
        ...     
        ...     # Partial derivatives of residual w.r.t. inputs
        ...     partials['x', 'a'] = x**2      # dR/da
        ...     partials['x', 'b'] = x         # dR/db
        ...     partials['x', 'c'] = 1.0       # dR/dc
        ...     
        ...     # Partial derivative of residual w.r.t. output
        ...     partials['x', 'x'] = 2*a*x + b # dR/dx

        For system of equations:

        >>> def residual_partials(self, inputs, outputs, partials):
        ...     x1, x2 = outputs['x1'], outputs['x2']
        ...     
        ...     # First residual: R1 = x1² + x2 - p1
        ...     partials['x1', 'p1'] = -1.0
        ...     partials['x1', 'x1'] = 2*x1
        ...     partials['x1', 'x2'] = 1.0
        ...     
        ...     # Second residual: R2 = x1 + x2² - p2
        ...     partials['x2', 'p2'] = -1.0
        ...     partials['x2', 'x1'] = 1.0
        ...     partials['x2', 'x2'] = 2*x2

        Notes
        -----
        - Partial derivatives should be computed at the current (inputs, outputs) point
        - All declared partial derivative pairs must be computed
        - Use setup_partials() to declare which partials will be computed
        - Partials are used for optimization, sensitivity analysis, and coupled solving
        - Consider finite differencing if analytical derivatives are too complex
        - The dR/doutputs terms are crucial for implicit system solution algorithms
        """
        raise NotImplementedError("residual_partials not implemented")

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        """
        Apply the linearized residual operator for matrix-free methods.

        This method implements the action of the residual Jacobian on vectors without
        explicitly forming the full Jacobian matrix. It supports both forward and reverse
        mode automatic differentiation, enabling efficient gradient computation for
        large-scale problems.

        The method computes:
        - Forward mode: d_residuals = J * [d_inputs; d_outputs]
        - Reverse mode: [d_inputs; d_outputs] += J^T * d_residuals

        Where J is the Jacobian matrix [dR/dinputs, dR/doutputs].

        Parameters
        ----------
        inputs : dict
            Dictionary of input values with variable names as keys
        outputs : dict
            Dictionary of output values with variable names as keys
        d_inputs : dict
            Dictionary of input perturbations/adjoint variables
        d_outputs : dict
            Dictionary of output perturbations/adjoint variables
        d_residuals : dict
            Dictionary of residual perturbations/adjoint variables
        mode : str
            Differentiation mode: 'fwd' for forward mode, 'rev' for reverse mode

        Examples
        --------
        For quadratic equation R = ax² + bx + c:

        >>> def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        ...     a, b, c = inputs['a'], inputs['b'], inputs['c']
        ...     x = outputs['x']
        ...     
        ...     if mode == 'fwd':
        ...         if 'x' in d_residuals:
        ...             # Forward: dR = (dR/da)*da + (dR/db)*db + (dR/dc)*dc + (dR/dx)*dx
        ...             if 'a' in d_inputs:
        ...                 d_residuals['x'] += x**2 * d_inputs['a']
        ...             if 'b' in d_inputs:
        ...                 d_residuals['x'] += x * d_inputs['b']
        ...             if 'c' in d_inputs:
        ...                 d_residuals['x'] += d_inputs['c']
        ...             if 'x' in d_outputs:
        ...                 d_residuals['x'] += (2*a*x + b) * d_outputs['x']
        ...     
        ...     elif mode == 'rev':
        ...         if 'x' in d_residuals:
        ...             # Reverse: accumulate adjoint contributions
        ...             if 'a' in d_inputs:
        ...                 d_inputs['a'] += x**2 * d_residuals['x']
        ...             if 'b' in d_inputs:
        ...                 d_inputs['b'] += x * d_residuals['x']
        ...             if 'c' in d_inputs:
        ...                 d_inputs['c'] += d_residuals['x']
        ...             if 'x' in d_outputs:
        ...                 d_outputs['x'] += (2*a*x + b) * d_residuals['x']

        For coupled system:

        >>> def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        ...     x1, x2 = outputs['x1'], outputs['x2']
        ...     
        ...     if mode == 'fwd':
        ...         # R1 = x1² + x2 - p1
        ...         if 'x1' in d_residuals:
        ...             if 'x1' in d_outputs:
        ...                 d_residuals['x1'] += 2*x1 * d_outputs['x1']
        ...             if 'x2' in d_outputs:
        ...                 d_residuals['x1'] += d_outputs['x2']
        ...             if 'p1' in d_inputs:
        ...                 d_residuals['x1'] -= d_inputs['p1']
        ...         
        ...         # R2 = x1 + x2² - p2
        ...         if 'x2' in d_residuals:
        ...             if 'x1' in d_outputs:
        ...                 d_residuals['x2'] += d_outputs['x1']
        ...             if 'x2' in d_outputs:
        ...                 d_residuals['x2'] += 2*x2 * d_outputs['x2']
        ...             if 'p2' in d_inputs:
        ...                 d_residuals['x2'] -= d_inputs['p2']
        ...     
        ...     elif mode == 'rev':
        ...         # Reverse mode: transpose operations
        ...         # ... (similar structure with += operations)

        Notes
        -----
        - This method enables matrix-free Newton and quasi-Newton methods
        - Essential for large-scale problems where storing full Jacobians is impractical
        - Forward mode: multiply Jacobian by vector (useful for directional derivatives)
        - Reverse mode: multiply Jacobian transpose by vector (useful for gradients)
        - Must be consistent with residual_partials() when implemented
        - Used by advanced optimization algorithms and coupled system solvers
        """
        raise NotImplementedError("apply_linear not implemented")

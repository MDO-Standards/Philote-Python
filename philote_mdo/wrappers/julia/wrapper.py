"""
Python discipline wrapper that calls Julia code via juliacall.

This allows using pure Julia disciplines with the proven Python gRPC server.
"""
import os
import numpy as np

# Import Philote disciplines
from philote_mdo.general.explicit_discipline import ExplicitDiscipline
from philote_mdo.general.implicit_discipline import ImplicitDiscipline

# Import juliacall
try:
    from juliacall import Main as jl
except ImportError as e:
    raise ImportError(
        f"Cannot import juliacall. Please install it with: pip install juliacall\n"
        f"Original error: {e}"
    )


class JuliaWrapperDiscipline(ExplicitDiscipline):
    """
    Python discipline that wraps a Julia Philote discipline.

    This uses juliacall to load and execute Julia code, while presenting
    a pure Python interface that works with the Philote-Python server.
    """

    def __init__(self, julia_file, julia_type, options=None):
        """
        Initialize with a Julia discipline.

        Args:
            julia_file: Path to Julia file containing the discipline
            julia_type: Name of the Julia type (e.g., "ParaboloidDiscipline")
            options: Optional dict of discipline options to set after initialization
        """
        super().__init__()

        self.julia_file = os.path.abspath(julia_file)
        self.julia_type = julia_type
        self.julia_discipline = None
        self.julia_metadata = None
        self._options = options or {}

        # Load Julia discipline
        self._load_julia_discipline()

    def _load_julia_discipline(self):
        """Load the Julia discipline using juliacall."""
        print(f"Loading Julia discipline from: {self.julia_file}")

        # Load the Philote.jl module
        philote_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        jl.seval(f'push!(LOAD_PATH, "{philote_dir}")')
        jl.seval('using Philote')

        # Load the discipline file
        if not os.path.exists(self.julia_file):
            raise FileNotFoundError(f"Julia file not found: {self.julia_file}")

        jl.seval(f'include("{self.julia_file}")')

        # Create discipline instance
        try:
            self.julia_discipline = jl.seval(f'{self.julia_type}()')
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate Julia type '{self.julia_type}'. "
                f"Make sure it exists in {self.julia_file}\n"
                f"Original error: {e}"
            )

        # Set options if provided
        if self._options:
            jl.seval('Philote.set_options!')(self.julia_discipline, jl.Dict(self._options))

        # Call setup!
        jl.seval('Philote.setup!')(self.julia_discipline)

        # Get metadata
        self.julia_metadata = jl.seval('Philote.get_metadata')(self.julia_discipline)

        print(f"✓ Julia discipline loaded: {self.julia_metadata.name}")

    def setup(self):
        """
        Setup the discipline - define inputs, outputs, and partials.

        This reads metadata from the Julia discipline and configures
        the Python discipline interface.
        """
        # Add inputs from Julia metadata
        for name, (shape, units) in self.julia_metadata.inputs.items():
            self.add_input(name, shape=tuple(shape), units=units)

        # Add outputs from Julia metadata
        for name, (shape, units) in self.julia_metadata.outputs.items():
            self.add_output(name, shape=tuple(shape), units=units)

        # Declare partials from Julia metadata
        for output_name, input_name in self.julia_metadata.partials:
            self.declare_partials(output_name, input_name)

    def compute(self, inputs, outputs):
        """
        Compute outputs from inputs by calling Julia discipline.

        Args:
            inputs: Dict of input arrays
            outputs: Dict to populate with output arrays
        """
        try:
            # Convert Python dict to Julia dict
            jl_inputs = jl.Dict(inputs)

            # Call Julia compute
            jl_outputs = jl.seval('Philote.compute')(self.julia_discipline, jl_inputs)

            # Convert Julia outputs back to Python and populate outputs dict
            for name in jl_outputs.keys():
                outputs[name] = np.array(jl_outputs[name])
        except Exception as e:
            raise RuntimeError(f"Error in Julia compute: {e}")

    def compute_partials(self, inputs, partials):
        """
        Compute partial derivatives by calling Julia discipline.

        Args:
            inputs: Dict of input arrays
            partials: Dict to populate with partial derivative arrays
        """
        try:
            # Convert Python dict to Julia dict
            jl_inputs = jl.Dict(inputs)

            # Call Julia compute_partials
            jl_partials = jl.seval('Philote.compute_partials')(self.julia_discipline, jl_inputs)

            # Convert Julia partials back to Python
            # Julia returns: Dict{output_name => Dict{input_name => jacobian}}
            for output_name in jl_partials.keys():
                for input_name in jl_partials[output_name].keys():
                    key = (output_name, input_name)
                    partials[key] = np.array(jl_partials[output_name][input_name])
        except Exception as e:
            raise RuntimeError(f"Error in Julia compute_partials: {e}")


class JuliaImplicitWrapperDiscipline(ImplicitDiscipline):
    """
    Python implicit discipline that wraps a Julia Philote implicit discipline.

    This uses juliacall to load and execute Julia code for implicit disciplines,
    while presenting a pure Python interface that works with the Philote-Python server.
    """

    def __init__(self, julia_file, julia_type, options=None):
        """
        Initialize with a Julia implicit discipline.

        Args:
            julia_file: Path to Julia file containing the discipline
            julia_type: Name of the Julia type (e.g., "MyImplicitDiscipline")
            options: Optional dict of discipline options to set after initialization
        """
        super().__init__()

        self.julia_file = os.path.abspath(julia_file)
        self.julia_type = julia_type
        self.julia_discipline = None
        self.julia_metadata = None
        self._options = options or {}

        # Load Julia discipline
        self._load_julia_discipline()

    def _load_julia_discipline(self):
        """Load the Julia implicit discipline using juliacall."""
        print(f"Loading Julia implicit discipline from: {self.julia_file}")

        # Load the Philote.jl module
        philote_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        jl.seval(f'push!(LOAD_PATH, "{philote_dir}")')
        jl.seval('using Philote')

        # Load the discipline file
        if not os.path.exists(self.julia_file):
            raise FileNotFoundError(f"Julia file not found: {self.julia_file}")

        jl.seval(f'include("{self.julia_file}")')

        # Create discipline instance
        try:
            self.julia_discipline = jl.seval(f'{self.julia_type}()')
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate Julia type '{self.julia_type}'. "
                f"Make sure it exists in {self.julia_file}\n"
                f"Original error: {e}"
            )

        # Set options if provided
        if self._options:
            jl.seval('Philote.set_options!')(self.julia_discipline, jl.Dict(self._options))

        # Call setup!
        jl.seval('Philote.setup!')(self.julia_discipline)

        # Get metadata
        self.julia_metadata = jl.seval('Philote.get_metadata')(self.julia_discipline)

        print(f"✓ Julia implicit discipline loaded: {self.julia_metadata.name}")

    def setup(self):
        """
        Setup the implicit discipline - define inputs, outputs, residuals, and partials.

        This reads metadata from the Julia discipline and configures
        the Python discipline interface.
        """
        # Add inputs from Julia metadata
        for name, (shape, units) in self.julia_metadata.inputs.items():
            self.add_input(name, shape=tuple(shape), units=units)

        # Add outputs from Julia metadata
        for name, (shape, units) in self.julia_metadata.outputs.items():
            self.add_output(name, shape=tuple(shape), units=units)

        # Add residuals from Julia metadata (implicit disciplines)
        for name, (shape, units) in self.julia_metadata.residuals.items():
            self.add_residual(name, shape=tuple(shape), units=units)

        # Declare partials from Julia metadata
        for output_name, input_name in self.julia_metadata.partials:
            self.declare_partials(output_name, input_name)

    def compute_residuals(self, inputs, outputs, residuals):
        """
        Compute residuals by calling Julia discipline.

        Args:
            inputs: Dict of input arrays
            outputs: Dict of output arrays
            residuals: Dict to populate with residual arrays
        """
        try:
            # Convert Python dicts to Julia dicts
            jl_inputs = jl.Dict(inputs)
            jl_outputs = jl.Dict(outputs)

            # Call Julia compute_residuals
            jl_residuals = jl.seval('Philote.compute_residuals')(
                self.julia_discipline, jl_inputs, jl_outputs
            )

            # Convert Julia residuals back to Python and populate residuals dict
            for name in jl_residuals.keys():
                residuals[name] = np.array(jl_residuals[name])
        except Exception as e:
            raise RuntimeError(f"Error in Julia compute_residuals: {e}")

    def solve_residuals(self, inputs, outputs):
        """
        Solve for outputs that drive residuals to zero by calling Julia discipline.

        Args:
            inputs: Dict of input arrays
            outputs: Dict of output arrays (modified in place)
        """
        try:
            # Convert Python dicts to Julia dicts
            jl_inputs = jl.Dict(inputs)
            jl_outputs = jl.Dict(outputs)

            # Call Julia solve_residuals (modifies outputs in place)
            jl.seval('Philote.solve_residuals')(
                self.julia_discipline, jl_inputs, jl_outputs
            )

            # Update Python outputs from Julia outputs
            for name in jl_outputs.keys():
                outputs[name] = np.array(jl_outputs[name])
        except Exception as e:
            raise RuntimeError(f"Error in Julia solve_residuals: {e}")

    def residual_partials(self, inputs, outputs, partials):
        """
        Compute residual partial derivatives by calling Julia discipline.

        Args:
            inputs: Dict of input arrays
            outputs: Dict of output arrays
            partials: Dict to populate with partial derivative arrays
        """
        try:
            # Convert Python dicts to Julia dicts
            jl_inputs = jl.Dict(inputs)
            jl_outputs = jl.Dict(outputs)

            # Call Julia residual_partials
            jl_partials = jl.seval('Philote.residual_partials')(
                self.julia_discipline, jl_inputs, jl_outputs
            )

            # Convert Julia partials back to Python
            # Julia returns: Dict{residual_name => Dict{variable_name => jacobian}}
            for residual_name in jl_partials.keys():
                for variable_name in jl_partials[residual_name].keys():
                    key = (residual_name, variable_name)
                    partials[key] = np.array(jl_partials[residual_name][variable_name])
        except Exception as e:
            raise RuntimeError(f"Error in Julia residual_partials: {e}")

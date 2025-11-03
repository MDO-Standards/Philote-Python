"""
Simple implicit discipline example: Quadratic equation solver.

This discipline solves the equation: a*x^2 + b*x + c = 0

Inputs:  a, b, c (coefficients)
Outputs: x (solution)
Residual: r = a*x^2 + b*x + c

This demonstrates implicit disciplines where the output must satisfy
a residual equation rather than being directly computed from inputs.
"""

using Philote

mutable struct QuadraticDiscipline <: Philote.ImplicitDiscipline
    tolerance::Float64

    function QuadraticDiscipline()
        new(1e-10)
    end
end

function Philote.setup!(discipline::QuadraticDiscipline)
    # Define inputs (coefficients of quadratic equation)
    Philote.add_input!(discipline, "a", [1], "unitless")
    Philote.add_input!(discipline, "b", [1], "unitless")
    Philote.add_input!(discipline, "c", [1], "unitless")

    # Define output (solution to equation)
    Philote.add_output!(discipline, "x", [1], "unitless")

    # Define residual (r = a*x^2 + b*x + c)
    Philote.add_residual!(discipline, "r", [1], "unitless")

    # Declare partial derivatives
    # ∂r/∂a = x^2
    Philote.declare_partials!(discipline, "r", "a")
    # ∂r/∂b = x
    Philote.declare_partials!(discipline, "r", "b")
    # ∂r/∂c = 1
    Philote.declare_partials!(discipline, "r", "c")
    # ∂r/∂x = 2*a*x + b
    Philote.declare_partials!(discipline, "r", "x")
end

function Philote.compute_residuals(discipline::QuadraticDiscipline,
                                   inputs::Dict{String,Array},
                                   outputs::Dict{String,Array})
    # Extract inputs
    a = inputs["a"][1]
    b = inputs["b"][1]
    c = inputs["c"][1]
    x = outputs["x"][1]

    # Compute residual: r = a*x^2 + b*x + c
    r = a * x^2 + b * x + c

    return Dict("r" => [r])
end

function Philote.solve_residuals(discipline::QuadraticDiscipline,
                                 inputs::Dict{String,Array},
                                 outputs::Dict{String,Array})
    # Extract coefficients
    a = inputs["a"][1]
    b = inputs["b"][1]
    c = inputs["c"][1]

    # Solve quadratic equation using quadratic formula
    # x = (-b ± sqrt(b^2 - 4ac)) / (2a)
    discriminant = b^2 - 4*a*c

    if discriminant < 0
        error("No real solution: discriminant is negative")
    end

    # Take the positive root (could make this configurable)
    x = (-b + sqrt(discriminant)) / (2*a)

    # Update output in place
    outputs["x"][1] = x
end

function Philote.residual_partials(discipline::QuadraticDiscipline,
                                   inputs::Dict{String,Array},
                                   outputs::Dict{String,Array})
    # Extract values
    a = inputs["a"][1]
    b = inputs["b"][1]
    c = inputs["c"][1]
    x = outputs["x"][1]

    # Compute partial derivatives of residual
    # r = a*x^2 + b*x + c

    # ∂r/∂a = x^2
    dr_da = x^2

    # ∂r/∂b = x
    dr_db = x

    # ∂r/∂c = 1
    dr_dc = 1.0

    # ∂r/∂x = 2*a*x + b
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

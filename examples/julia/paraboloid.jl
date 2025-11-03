"""
    Paraboloid Discipline Example

This example implements the paraboloid function:
    f(x, y) = (x - 3)^2 + x*y + (y + 4)^2 - 3

This is the same example used in Philote-Cpp for consistency.
"""

# Load the Philote module from the parent directory
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Philote

"""
    ParaboloidDiscipline <: ExplicitDiscipline

Example explicit discipline computing a paraboloid function.

Options:
- scale_factor (float): Scaling factor applied to output (default: 1.0)
- offset (float): Offset added to output (default: 0.0)
"""
mutable struct ParaboloidDiscipline <: Philote.ExplicitDiscipline
    scale_factor::Float64
    offset::Float64

    function ParaboloidDiscipline()
        new(1.0, 0.0)
    end
end

"""
    setup!(discipline::ParaboloidDiscipline)

Declare inputs, outputs, options, and partials for the paraboloid discipline.
"""
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

"""
    compute(discipline::ParaboloidDiscipline, inputs::Dict{String, Array{Float64}})

Compute the paraboloid function output.

# Formula
f(x, y) = scale_factor * [(x - 3)^2 + x*y + (y + 4)^2 - 3] + offset
"""
function Philote.compute(discipline::ParaboloidDiscipline,
                        inputs::Dict{String, <:AbstractArray{Float64}})
    # Extract scalar inputs
    x = inputs["x"][1]
    y = inputs["y"][1]

    # Compute paraboloid function
    f_xy = (x - 3.0)^2 + x * y + (y + 4.0)^2 - 3.0

    # Apply scaling and offset
    f_xy = discipline.scale_factor * f_xy + discipline.offset

    # Return outputs as dictionary
    return Dict("f_xy" => [f_xy])
end

"""
    compute_partials(discipline::ParaboloidDiscipline, inputs::Dict{String, Array{Float64}})

Compute analytical gradients of the paraboloid function.

# Derivatives
∂f/∂x = scale_factor * [2(x - 3) + y]
∂f/∂y = scale_factor * [2(y + 4) + x]
"""
function Philote.compute_partials(discipline::ParaboloidDiscipline,
                                  inputs::Dict{String, <:AbstractArray{Float64}})
    # Extract scalar inputs
    x = inputs["x"][1]
    y = inputs["y"][1]

    # Compute partials
    df_dx = discipline.scale_factor * (2.0 * (x - 3.0) + y)
    df_dy = discipline.scale_factor * (2.0 * (y + 4.0) + x)

    # Return as nested dictionary
    return Dict(
        "f_xy" => Dict(
            "x" => [df_dx],
            "y" => [df_dy]
        )
    )
end

"""
    set_options!(discipline::ParaboloidDiscipline, options::Dict{String, <:Any})

Set discipline options from a dictionary.
"""
function Philote.set_options!(discipline::ParaboloidDiscipline, options::Dict{String, <:Any})
    if haskey(options, "scale_factor")
        discipline.scale_factor = Float64(options["scale_factor"])
    end
    if haskey(options, "offset")
        discipline.offset = Float64(options["offset"])
    end
end

# Example usage when run as a standalone script
if abspath(PROGRAM_FILE) == @__FILE__
    println("Paraboloid Discipline Example")
    println("=" ^ 50)

    # Create discipline instance
    discipline = ParaboloidDiscipline()

    # Setup discipline
    Philote.setup!(discipline)

    # Print metadata
    meta = Philote.get_metadata(discipline)
    println("Discipline: $(meta.name) v$(meta.version)")
    println("\nInputs:")
    for (name, (shape, units)) in meta.inputs
        println("  $name: shape=$(shape), units=$units")
    end
    println("\nOutputs:")
    for (name, (shape, units)) in meta.outputs
        println("  $name: shape=$(shape), units=$units")
    end
    println("\nPartials:")
    for (output, input) in meta.partials
        println("  ∂$output/∂$input")
    end

    # Test computation
    println("\n" * "=" ^ 50)
    println("Test Evaluation")
    println("=" ^ 50)

    inputs = Dict("x" => [1.0], "y" => [2.0])
    println("Inputs: x=$(inputs["x"][1]), y=$(inputs["y"][1])")

    outputs = Philote.compute(discipline, inputs)
    println("Output: f_xy=$(outputs["f_xy"][1])")

    partials = Philote.compute_partials(discipline, inputs)
    println("Gradient: ∂f/∂x=$(partials["f_xy"]["x"][1]), ∂f/∂y=$(partials["f_xy"]["y"][1])")

    # Test with options
    println("\n" * "=" ^ 50)
    println("Test with Options")
    println("=" ^ 50)

    Philote.set_options!(discipline, Dict("scale_factor" => 2.0, "offset" => 10.0))
    println("Options: scale_factor=$(discipline.scale_factor), offset=$(discipline.offset)")

    outputs = Philote.compute(discipline, inputs)
    println("Output: f_xy=$(outputs["f_xy"][1])")

    partials = Philote.compute_partials(discipline, inputs)
    println("Gradient: ∂f/∂x=$(partials["f_xy"]["x"][1]), ∂f/∂y=$(partials["f_xy"]["y"][1])")
end

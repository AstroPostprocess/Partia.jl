"""CUDA implementations of Partia grid construction and coordinate materialization."""
module Grids
using Partia
using CUDA

# Grid construction
include(joinpath(@__DIR__, "PointSamples", "PointSamples.jl"))
include(joinpath(@__DIR__, "PointSamples", "geometry_kernel.jl"))
include(joinpath(@__DIR__, "PointSamples", "build.jl"))
include(joinpath(@__DIR__, "PointSamples", "update.jl"))
include(joinpath(@__DIR__, "LineSamples", "LineSamples.jl"))
include(joinpath(@__DIR__, "LineSamples", "geometry_kernel.jl"))
include(joinpath(@__DIR__, "LineSamples", "build.jl"))
include(joinpath(@__DIR__, "LineSamples", "update.jl"))
include(joinpath(@__DIR__, "StructuredGrid", "coordinate_grid_kernel.jl"))
include(joinpath(@__DIR__, "StructuredGrid", "StructuredGrid.jl"))


# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end

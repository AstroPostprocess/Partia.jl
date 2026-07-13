module KernelInterpolation
using Partia
using Metal
using StaticArrays
using UnsignedRadixSorts

# Interpolation setup
include(joinpath(@__DIR__, "interpolation_setup", "InterpolationInput.jl"))
include(joinpath(@__DIR__, "interpolation_setup", "InterpolationSmoothingVolumeInput.jl"))

# Grid interpolation
include(joinpath(@__DIR__, "grid_interpolation", "kernels", "PointSamples_kernel.jl"))
include(joinpath(@__DIR__, "grid_interpolation", "kernels", "LineSamples_kernel.jl"))
include(joinpath(@__DIR__, "grid_interpolation", "drivers", "PointSamples_driver.jl"))
include(joinpath(@__DIR__, "grid_interpolation", "drivers", "LineSamples_driver.jl"))

# Export functions, macros, constants, and types.
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end

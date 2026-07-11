"""
KernelInterpolation

Core infrastructure for kernel-based interpolation of SPH data in Partia.

This module implements the full interpolation pipeline used to map particle-based
SPH quantities onto arbitrary points or grids. It is designed to be:

- Numerically explicit and allocation-aware
- Compatible with GPU execution via `Adapt.jl`
- Structured around LBVH-based neighbor traversal

# Scope and Responsibilities

The module provides:

## Kernel Definitions
- Spline kernels (M4, M5, M6)
- Wendland kernels (C2, C4, C6)
- line integrated kernels

Kernel implementations are located under:
- `kernel_function/`
- `kernel_function/kernels/`

## Line Integration Tables
- Precomputed tables for line-integrated kernels
- Used by projected / column-integrated quantities

Implemented under:
- `table/`

## Interpolation Framework
- Strategy and catalog abstractions for interpolation modes
- Strongly-typed, GPU-safe interpolation input definitions

Implemented under:
- `interpolation_setup/`

## Single-Point Interpolation
- Scalar interpolation
- Line-integrated scalar interpolation
- Gradient, divergence, and curl evaluation
- General interpolation kernels

All single-point interpolation routines are implemented using
LBVH-based neighbor traversal and accumulation, under:
- `single_point_interpolation/`

## Grid-Based Interpolation
- Sampling SPH data onto structured grids

Implemented under:
- `grid_interpolation/`
"""
module KernelInterpolation
using .Threads
using StaticArrays
using Adapt

using Partia.Grids
using Partia.LinearBoundingVolumeHierarchy

# KernelInterpolation
include(joinpath(@__DIR__, "table", "line_integrated_kernel_tables.jl"))
## Kernels
include(joinpath(@__DIR__, "kernel_function", "kernel.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "M4_spline.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "M5_spline.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "M6_spline.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "C2_Wendland.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "C4_Wendland.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "C6_Wendland.jl"))
include(joinpath(@__DIR__, "kernel_function", "line_integrated_kernel.jl"))

## Execution backends
include(joinpath(@__DIR__, "ExecutionBackend", "AbstractExecutionBackend.jl"))

## Single point interpolation
include(joinpath(@__DIR__, "interpolation_setup", "InterpolationStrategy.jl"))
include(joinpath(@__DIR__, "interpolation_setup", "InterpolationCatalog.jl"))
include(joinpath(@__DIR__, "interpolation_setup", "AbstractInterpolationInput.jl"))
include(joinpath(@__DIR__, "interpolation_setup", "InterpolationInput.jl"))
include(joinpath(@__DIR__, "interpolation_setup", "InterpolationSmoothingVolumeInput.jl"))
include(joinpath(@__DIR__, "interpolation_setup", "constructor.jl"))

### LBVH Traversal
#### Point interpolations
include(joinpath(@__DIR__, "single_point_interpolation", "gather_interpolation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "scatter_interpolation.jl"))

#### Line integrated interpolations
include(joinpath(@__DIR__, "line_integrated_interpolation", "line_integrated_scalar_interpolation.jl"))

## Grid interpolation
### Kernels
include(joinpath(@__DIR__, "grid_interpolation", "kernels", "point_samples_kernel.jl"))
include(joinpath(@__DIR__, "grid_interpolation", "kernels", "line_samples_kernel.jl"))

### Drivers
include(joinpath(@__DIR__, "grid_interpolation", "drivers", "point_samples_driver.jl"))
include(joinpath(@__DIR__, "grid_interpolation", "drivers", "line_samples_driver.jl"))

### Wrappers
include(joinpath(@__DIR__, "grid_interpolation", "wrappers", "wrapper_utils.jl"))
include(joinpath(@__DIR__, "grid_interpolation", "wrappers", "point_samples_wrapper.jl"))
include(joinpath(@__DIR__, "grid_interpolation", "wrappers", "line_samples_wrapper.jl"))
include(joinpath(@__DIR__, "grid_interpolation", "wrappers", "structured_grid_wrapper.jl"))


# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end

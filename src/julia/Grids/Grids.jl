"""
Grids

Grid data abstractions and representations for Partia.

This module defines the grid-side data structures used for sampling,
storing, and manipulating gridded quantities derived from SPH data.
It provides a unified interface for both structured and unstructured
grid representations, together with dataset-level containers.

# Scope and Responsibilities

The module provides:

## Coordinate System Definitions
- Flags and utilities for identifying grid coordinate systems
  (e.g. Cartesian, cylindrical, spherical)

Implemented in:
- `coordinate/coordinate.jl`
- `coordinate/cartesian.jl`
- `coordinate/polar.jl`
- `coordinate/cylindrical.jl`

## Core Grid Abstractions
- `AbstractGrid`, the common interface for all grid types
- `AbstractSamples`, the common interface for sample-based grid types
- `PointSamples`, a flexible sample representation with explicit coordinates
- `LineSamples`, a flexible line-sample representation with explicit origins and directions
- `StructuredGrid`, a regular grid with implicit topology

Implemented in:
- `abstract/AbstractGrid.jl`
- `abstract/AbstractSamples.jl`
- `PointSamples/`
- `LineSamples/`
- `StructuredGrid/StructuredGrid.jl`

## Grid Transformations
- Conversion utilities between `StructuredGrid` and `PointSamples`
- Used to bridge regular grids and more general representations

Implemented in:
- `transform/transform.jl`

## Grid Dataset Containers
- `GridBundle`, a lightweight container for grouped grid objects
- `GridDataset`, a dataset-level abstraction for gridded fields

Implemented in:
- `griddataset/GridBundle.jl`
- `griddataset/GridDataset.jl`
"""
module Grids
using .Threads
using Statistics
using Adapt
import ..Tools
import ..Tools: build!, update!
using Partia.Tools: _cylin2cart, _sph2cart
using Partia.Frames

# Flag of coordinate system
include(joinpath(@__DIR__, "coordinate", "coordinate.jl"))
include(joinpath(@__DIR__, "coordinate", "cartesian.jl"))
include(joinpath(@__DIR__, "coordinate", "polar.jl"))
include(joinpath(@__DIR__, "coordinate", "cylindrical.jl"))

# AbstractBeamModel
include(joinpath(@__DIR__, "abstract", "AbstractBeamModel.jl"))

# AbstractGrid
include(joinpath(@__DIR__, "abstract", "AbstractGrid.jl"))

# AbstractSamples
include(joinpath(@__DIR__, "abstract", "AbstractSamples.jl"))

# PointSamples
include(joinpath(@__DIR__, "PointSamples", "PointSamples.jl"))
include(joinpath(@__DIR__, "PointSamples", "build.jl"))
include(joinpath(@__DIR__, "PointSamples", "update.jl"))

# LineSamples
include(joinpath(@__DIR__, "LineSamples", "LineSamples.jl"))
include(joinpath(@__DIR__, "LineSamples", "build.jl"))
include(joinpath(@__DIR__, "LineSamples", "update.jl"))

# StructuredGrid
include(joinpath(@__DIR__, "StructuredGrid", "StructuredGrid.jl"))

# Transfromation between StructuredGrid and PointSamples
include(joinpath(@__DIR__, "transform", "transform.jl"))

# GridBundle
include(joinpath(@__DIR__,  "griddataset", "GridBundle.jl"))

# GridDataset
include(joinpath(@__DIR__,  "griddataset", "GridDataset.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end

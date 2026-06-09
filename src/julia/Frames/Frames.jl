"""
Frames

Reference-frame utilities for constructing oriented sampling planes and applying
frame-local rotations and translations.

This module defines a mutable `Frame` with a global position, an initial local
basis, and an accumulated orientation quaternion. It also provides coordinate
dispatch tags for interpreting translations in either global or frame-local
coordinates.

# Scope and Responsibilities

The module provides:

## Frame Representation
- `Frame`, storing position, initial basis vectors, and orientation quaternion
- Accessors for the current position and rotated basis directions

Implemented in:
- `struct/Frame.jl`

## Coordinate Dispatch Tags
- `GlobalCoordinates`
- `LocalCoordinates`

Implemented in:
- `struct/AbstractFrameCoordinates.jl`

## Frame Operations
- `rotate!`, extending `LinearAlgebra.rotate!` for `Frame`
- `translate!`, updating frame position in global or local coordinates

Implemented in:
- `operations/rotation.jl`
- `operations/translation.jl`
"""
module Frames
using Quaternions
using StaticArrays
using LinearAlgebra: norm, dot, cross
import LinearAlgebra: rotate!

export rotate!

# Frame
include(joinpath(@__DIR__, "struct", "Frame.jl"))

# Coordinate indicators
include(joinpath(@__DIR__, "struct", "AbstractFrameCoordinates.jl"))

# Rotation
include(joinpath(@__DIR__, "operations", "rotation.jl"))

# Translation
include(joinpath(@__DIR__, "operations", "translation.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end

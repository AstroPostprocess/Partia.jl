"""
LinearBoundingVolumeHierarchy

Provides Morton encoding, binary radix tree construction, and linear bounding
volume hierarchy (LinearBVH) query routines for SPH data.

Implementations live under the `MortonEncoding/`, `BinaryRadixTree/`,
`AxisAlignedBoundingBox/`, `BoxScale/`, and `LinearBVH/` directories.

    by Wei-Shan Su,
    May 4, 2026
"""
module LinearBoundingVolumeHierarchy
using .Threads
using Statistics
using StaticArrays
using Adapt

# Morton encoding
include(joinpath(@__DIR__, "MortonEncoding", "MortonEncoding.jl"))
include(joinpath(@__DIR__, "MortonEncoding", "toolbox.jl"))

# Binary radix tree
include(joinpath(@__DIR__, "BinaryRadixTree", "toolbox.jl"))
include(joinpath(@__DIR__, "BinaryRadixTree", "BinaryRadixTree.jl"))

# Shared neighbor selection container
include(joinpath(@__DIR__, "NeighborSelection.jl"))

# Axis-aligned bounding boxes
include(joinpath(@__DIR__, "AxisAlignedBoundingBox", "AABB.jl"))
include(joinpath(@__DIR__, "AxisAlignedBoundingBox", "toolbox.jl"))

# Box scale descriptors
include(joinpath(@__DIR__, "BoxScale", "BoxScale.jl"))

# Linear bounding volume hierarchy (LinearBVH)
include(joinpath(@__DIR__, "LinearBVH", "LinearBVH.jl"))
include(joinpath(@__DIR__, "LinearBVH", "toolbox.jl"))
include(joinpath(@__DIR__, "LinearBVH", "LinearBVHTraversalMacros.jl"))
include(joinpath(@__DIR__, "LinearBVH", "LinearBVHQueries.jl"))

# Export functions, macros, constants, and types.
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end

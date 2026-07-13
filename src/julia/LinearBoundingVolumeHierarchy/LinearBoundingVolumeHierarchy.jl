"""
LinearBoundingVolumeHierarchy

Provides Morton encoding, linear bounding volume hierarchy construction, and
LinearBVH query routines for SPH data.

Implementations live under the `MortonEncoding/`,
`AxisAlignedBoundingBox/` and `LinearBVH/` directories.

    by Wei-Shan Su,
    July 12, 2026
"""
module LinearBoundingVolumeHierarchy
using .Threads
using Statistics
using StaticArrays
using Adapt
using Atomix
using UnsignedRadixSorts

# Morton encoding
include(joinpath(@__DIR__, "MortonEncoding", "CoordinateQuantization", "quantization_scale.jl"))
include(joinpath(@__DIR__, "MortonEncoding", "BitInterleaving", "bit_expansion.jl"))
include(joinpath(@__DIR__, "MortonEncoding", "BitInterleaving", "morton_code.jl"))
include(joinpath(@__DIR__, "MortonEncoding", "morton_encoding_kernel.jl"))
include(joinpath(@__DIR__, "MortonEncoding", "MortonEncoding.jl"))
include(joinpath(@__DIR__, "MortonOrdering", "morton_ordering.jl"))
include(joinpath(@__DIR__, "MortonEncoding", "build.jl"))

# Shared neighbor selection container
include(joinpath(@__DIR__, "NeighborSelection.jl"))

# Axis-aligned bounding boxes
include(joinpath(@__DIR__, "AxisAlignedBoundingBox", "AABB.jl"))
include(joinpath(@__DIR__, "AxisAlignedBoundingBox", "toolbox.jl"))

# Linear bounding volume hierarchy (LinearBVH)
include(joinpath(@__DIR__, "LinearBVH", "Topology", "node_indexing.jl"))
include(joinpath(@__DIR__, "LinearBVH", "Topology", "longest_common_prefix.jl"))
include(joinpath(@__DIR__, "LinearBVH", "initialize_leaf_node.jl"))
include(joinpath(@__DIR__, "LinearBVH", "LinearBVH.jl"))
include(joinpath(@__DIR__, "LinearBVH", "ascend_from_leaf.jl"))
include(joinpath(@__DIR__, "LinearBVH", "build.jl"))
include(joinpath(@__DIR__, "LinearBVH", "linear_bvh_traversal_macros.jl"))
include(joinpath(@__DIR__, "LinearBVH", "linear_bvh_queries.jl"))

# Export functions, macros, constants, and types.
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end

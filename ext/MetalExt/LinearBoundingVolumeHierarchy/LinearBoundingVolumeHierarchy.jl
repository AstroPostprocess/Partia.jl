"""Metal Morton encoding, sorting, and linear-BVH construction methods."""
module LinearBoundingVolumeHierarchy
using Partia
using Metal
using UnsignedRadixSorts
using ..Tools: _metal_weak_cas_rendezvous

# Morton encoding
include(joinpath(@__DIR__, "MortonEncoding", "morton_encoding_kernel.jl"))
include(joinpath(@__DIR__, "MortonEncoding", "MortonEncoding.jl"))

# Sorting
include(joinpath(@__DIR__, "MortonOrdering", "morton_ordering.jl"))
include(joinpath(@__DIR__, "MortonEncoding", "build.jl"))
include(joinpath(@__DIR__, "MortonEncoding", "update.jl"))

# Linear bounding volume hierarchy (LinearBVH)
include(joinpath(@__DIR__, "LinearBVH", "initialize_leaf_node.jl"))
include(joinpath(@__DIR__, "LinearBVH", "LinearBVH.jl"))
include(joinpath(@__DIR__, "LinearBVH", "ascend_from_leaf.jl"))
include(joinpath(@__DIR__, "LinearBVH", "build.jl"))
include(joinpath(@__DIR__, "LinearBVH", "update.jl"))

# Export functions, macros, constants, and types.
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end

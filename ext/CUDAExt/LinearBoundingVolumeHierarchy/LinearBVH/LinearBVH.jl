######################################################################################

# Linear bounding volume hierarchy constructors for CUDA.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
"""
    LinearBVH(enc, scale,
              leaf_min, leaf_max,
              ::Val{NBlocks}=Val(256),
              ::Val{ThreadsPerBlock}=Val(256))
    LinearBVH(enc, scale,
              ::Val{NBlocks}=Val(256),
              ::Val{ThreadsPerBlock}=Val(256))

Construct a CUDA linear bounding volume hierarchy from Morton-sorted leaves.
The explicit form accepts a minimum and maximum coordinate vector for every
dimension. The shorter form treats each encoded coordinate as a point AABB.

The topology is built bottom-up by CUDA threads rendezvousing at split
positions. Internal AABBs and maximum scale values are merged as each subtree
becomes complete.

# Parameters
- `enc`: Morton-sorted codes and coordinates.
- `scale`: Per-leaf scale values in Morton order.
- `leaf_min`, `leaf_max`: Per-axis leaf bounds in Morton order.
- `NBlocks`: Number of CUDA blocks used by the construction kernels.
- `ThreadsPerBlock`: Number of threads in each CUDA block.

# Returns
A unified-node `LinearBVH` stored entirely in CUDA arrays.
"""
function Partia.LinearBVH(enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, scale :: CuVector{TF}, leaf_min :: NTuple{D, CuVector{TF}}, leaf_max :: NTuple{D, CuVector{TF}}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned}
    # All leaf attributes must already follow the Morton ordering in enc.
    codes = enc.codes
    n = length(codes)
    length(scale) == n || throw(DimensionMismatch("scale and enc.codes must have identical lengths"))
    all(length(v) == n for v in leaf_min) || throw(DimensionMismatch("leaf_min and enc.codes must have identical lengths"))
    all(length(v) == n for v in leaf_max) || throw(DimensionMismatch("leaf_max and enc.codes must have identical lengths"))
    n > 0 || throw(ArgumentError("LinearBVH requires at least one leaf"))
    n <= (typemax(Int32) ÷ 2) + 1 || throw(ArgumentError("leaf count exceeds Int32 node capacity"))

    # Unified IDs 1:(n-1) are internal nodes and n:(2n-1) are leaves.
    n_internal = n - 1
    total_length = 2n - 1
    # Allocate topology, AABB, and hierarchical scale storage on the device.
    left = CuVector{Int32}(undef, n_internal)
    escape = CUDA.zeros(Int32, total_length)
    aabb = Partia.AABB{D, TF, CuVector{TF}}(ntuple(_ -> CuVector{TF}(undef, total_length), D), ntuple(_ -> CuVector{TF}(undef, total_length), D))
    unified_scale = CuVector{TF}(undef, total_length)

    lbvh = Partia.LinearBVH{D, TF, CuVector{TF}, CuVector{Int32}}(n, left, escape, aabb, unified_scale)
    store = CuVector{Int32}(undef, n_internal)
    return Partia.build!(lbvh, store, enc, scale, leaf_min, leaf_max, Val(NBlocks), Val(ThreadsPerBlock))
end

function Partia.LinearBVH(enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, scale :: CuVector{TF}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned}
    # Point leaves use the Morton-sorted coordinates as both AABB endpoints.
    return Partia.LinearBVH(enc, scale, enc.coord, enc.coord, Val(NBlocks), Val(ThreadsPerBlock))
end

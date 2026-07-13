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
function Partia.LinearBVH(enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, scale :: CuVector{TF}, leaf_min :: NTuple{D, CuVector{TF}}, leaf_max :: NTuple{D, CuVector{TF}}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, TF <: AbstractFloat, NBlocks, ThreadsPerBlock, TI <: Unsigned}
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

    # Populate the leaf section before any parent thread reads it.
    @cuda threads=ThreadsPerBlock blocks=NBlocks Partia.LinearBoundingVolumeHierarchy._initialize_leaf_node!(unified_scale, aabb, scale, leaf_min, leaf_max, n_internal)
    # CUDA launches on the same stream are ordered, so the construction kernel
    # below observes the initialized leaf section without a host-side barrier.
    lbvh = Partia.LinearBVH{D, TF, CuVector{TF}, CuVector{Int32}}(n, left, escape, aabb, unified_scale)

    if n_internal > 0
        # A zero slot means no child subtree has reached this split yet. The
        # second child atomically retrieves the first child's boundary and
        # continues upward with the completed parent subtree.
        store = CUDA.zeros(Int32, n_internal)
        @cuda threads=ThreadsPerBlock blocks=NBlocks Partia.LinearBoundingVolumeHierarchy._ascend_from_leaf!(lbvh, store, codes)
        CUDA.synchronize()
    end
    return lbvh
end

function Partia.LinearBVH(enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, scale :: CuVector{TF}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, TF <: AbstractFloat, NBlocks, ThreadsPerBlock, TI <: Unsigned}
    # Point leaves use the Morton-sorted coordinates as both AABB endpoints.
    return Partia.LinearBVH(enc, scale, enc.coord, enc.coord, Val(NBlocks), Val(ThreadsPerBlock))
end

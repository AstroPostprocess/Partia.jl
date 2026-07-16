######################################################################################

# Reusable linear bounding volume hierarchy builds for CUDA.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
"""
    build!(lbvh, store, enc, scale, leaf_min, leaf_max,
           ::Val{NBlocks}=Val(256), ::Val{ThreadsPerBlock}=Val(256))
    build!(lbvh, store, enc, scale,
           ::Val{NBlocks}=Val(256), ::Val{ThreadsPerBlock}=Val(256))

Rebuild a preallocated CUDA `LinearBVH` and reusable rendezvous `store` from
Morton-sorted leaves. The explicit form uses the supplied leaf bounds; the
shorter form uses point AABBs from `enc.coord`. Construction is synchronized
before this function returns.

# Parameters
- `lbvh`: Preallocated CUDA hierarchy with `nleaf == length(enc.codes)`.
- `store`: Reusable `CuVector{Int32}` rendezvous storage of length `nleaf - 1`.
- `enc`: CUDA Morton encoding already sorted by `sort_by_morton!`.
- `scale`: Per-leaf scale values in the same Morton order as `enc`.
- `leaf_min`, `leaf_max`: Per-axis leaf bounds in that same order.
- `NBlocks`: Number of CUDA blocks used by the construction kernels. Defaults
  to 256.
- `ThreadsPerBlock`: Number of threads per CUDA block. Defaults to 256.

# Returns
- `nothing`: `lbvh` and `store` are updated in place.
"""
function Partia.build!(lbvh :: LinearBVH{D, TF, CuVector{TF}, CuVector{Int32}}, store :: CuVector{Int32}, enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, scale :: CuVector{TF}, leaf_min :: NTuple{D, CuVector{TF}}, leaf_max :: NTuple{D, CuVector{TF}}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned}
    codes = enc.codes
    n = length(codes)
    n == nleaf(lbvh) || throw(DimensionMismatch("nleaf(lbvh) and enc.codes must have identical lengths"))
    length(store) == n - 1 || throw(DimensionMismatch("store must have length nleaf(lbvh) - 1"))
    length(scale) == n || throw(DimensionMismatch("scale and enc.codes must have identical lengths"))
    all(length(v) == n for v in leaf_min) || throw(DimensionMismatch("leaf_min and enc.codes must have identical lengths"))
    all(length(v) == n for v in leaf_max) || throw(DimensionMismatch("leaf_max and enc.codes must have identical lengths"))
    Partia.Tools._issorted(codes) || throw(ArgumentError("LinearBVH: enc.codes must be sorted in nondecreasing order."))

    n_internal = n - 1
    fill!(store, zero(Int32))
    @cuda threads=ThreadsPerBlock blocks=NBlocks Partia.LinearBoundingVolumeHierarchy._initialize_leaf_node!(lbvh.scale, lbvh.aabb, scale, leaf_min, leaf_max, n_internal)

    if n_internal > 0
        # Same-stream launch ordering makes the initialized leaves visible to
        # the bottom-up topology construction without an intermediate barrier.
        @cuda threads=ThreadsPerBlock blocks=NBlocks Partia.LinearBoundingVolumeHierarchy._ascend_from_leaf!(lbvh, store, codes)
    end
    CUDA.synchronize()
    return nothing
end

function Partia.build!(lbvh :: LinearBVH{D, TF, CuVector{TF}, CuVector{Int32}}, store :: CuVector{Int32}, enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, scale :: CuVector{TF}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned}
    Partia.build!(lbvh, store, enc, scale, enc.coord, enc.coord, Val(NBlocks), Val(ThreadsPerBlock))
    return nothing
end

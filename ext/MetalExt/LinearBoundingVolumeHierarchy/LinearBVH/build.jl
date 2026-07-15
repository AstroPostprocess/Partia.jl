######################################################################################

# Reusable linear bounding volume hierarchy builds for Metal.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
"""
    build!(lbvh, store, enc, scale, leaf_min, leaf_max,
           ::Val{ThreadsPerGroup}=Val(256))
    build!(lbvh, store, enc, scale, ::Val{ThreadsPerGroup}=Val(256))

Rebuild a preallocated Metal `LinearBVH` and reusable rendezvous `store` from
Morton-sorted leaves. The explicit form uses the supplied leaf bounds; the
shorter form uses point AABBs from `enc.coord`. Construction is synchronized
before this function returns.

# Parameters
- `lbvh`: Preallocated Metal hierarchy with `nleaf == length(enc.codes)`.
- `store`: Reusable `MtlVector{Int32}` rendezvous storage of length `nleaf - 1`.
- `enc`: Metal Morton encoding already sorted by `sort_by_morton!`.
- `scale`: Per-leaf scale values in the same Morton order as `enc`.
- `leaf_min`, `leaf_max`: Per-axis leaf bounds in that same order.
- `ThreadsPerGroup`: Number of threads per Metal threadgroup. Defaults to 256.

# Returns
- `nothing`: `lbvh` and `store` are updated in place.
"""
function Partia.build!(lbvh :: LinearBVH{D, Float32, MtlVector{Float32}, MtlVector{Int32}}, store :: MtlVector{Int32}, enc :: MortonEncoding{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, scale :: MtlVector{Float32}, leaf_min :: NTuple{D, MtlVector{Float32}}, leaf_max :: NTuple{D, MtlVector{Float32}}, :: Val{ThreadsPerGroup} = Val(256)) where {D, ThreadsPerGroup, TI <: Unsigned}
    codes = enc.codes
    n = length(codes)
    n == nleaf(lbvh) || throw(DimensionMismatch("nleaf(lbvh) and enc.codes must have identical lengths"))
    length(store) == n - 1 || throw(DimensionMismatch("store must have length nleaf(lbvh) - 1"))
    length(scale) == n || throw(DimensionMismatch("scale and enc.codes must have identical lengths"))
    @inbounds for d in 1:D
        length(leaf_min[d]) == n || throw(DimensionMismatch("leaf_min[$d] and enc.codes must have identical lengths"))
        length(leaf_max[d]) == n || throw(DimensionMismatch("leaf_max[$d] and enc.codes must have identical lengths"))
    end
    Partia.Tools._issorted(codes) || throw(ArgumentError("LinearBVH: enc.codes must be sorted in nondecreasing order."))

    n_internal = n - 1
    fill!(store, zero(Int32))
    @metal threads=(ThreadsPerGroup,) groups=(cld(n, ThreadsPerGroup),) Partia.LinearBoundingVolumeHierarchy._initialize_leaf_node!(lbvh.scale, lbvh.aabb, scale, leaf_min, leaf_max, n_internal)

    if n_internal > 0
        # Same-queue command ordering removes the need for an intermediate
        # host-side synchronization between leaf initialization and ascent.
        @metal threads=(ThreadsPerGroup,) groups=(cld(n, ThreadsPerGroup),) Partia.LinearBoundingVolumeHierarchy._ascend_from_leaf!(lbvh, store, codes)
    end
    Metal.synchronize()
    return nothing
end

function Partia.build!(lbvh :: LinearBVH{D, Float32, MtlVector{Float32}, MtlVector{Int32}}, store :: MtlVector{Int32}, enc :: MortonEncoding{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, scale :: MtlVector{Float32}, :: Val{ThreadsPerGroup} = Val(256)) where {D, ThreadsPerGroup, TI <: Unsigned}
    Partia.build!(lbvh, store, enc, scale, enc.coord, enc.coord, Val(ThreadsPerGroup))
    return nothing
end

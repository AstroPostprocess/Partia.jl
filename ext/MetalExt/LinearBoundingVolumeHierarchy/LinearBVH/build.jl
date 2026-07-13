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
Morton-sorted leaves. The shorter form uses point AABBs from `enc.coord`.

# Returns
- `lbvh`: The rebuilt hierarchy after Metal construction has completed.
"""
function Partia.build!(lbvh :: LinearBVH{D, Float32, MtlVector{Float32}, MtlVector{Int32}}, store :: MtlVector{Int32}, enc :: MortonEncoding{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, scale :: MtlVector{Float32}, leaf_min :: NTuple{D, MtlVector{Float32}}, leaf_max :: NTuple{D, MtlVector{Float32}}, :: Val{ThreadsPerGroup} = Val(256)) where {D, ThreadsPerGroup, TI <: Unsigned}
    codes = enc.codes
    n = length(codes)
    n == lbvh.nleaf || throw(DimensionMismatch("lbvh.nleaf and enc.codes must have identical lengths"))
    length(store) == n - 1 || throw(DimensionMismatch("store must have length lbvh.nleaf - 1"))
    length(scale) == n || throw(DimensionMismatch("scale and enc.codes must have identical lengths"))
    @inbounds for d in 1:D
        length(leaf_min[d]) == n || throw(DimensionMismatch("leaf_min[$d] and enc.codes must have identical lengths"))
        length(leaf_max[d]) == n || throw(DimensionMismatch("leaf_max[$d] and enc.codes must have identical lengths"))
    end

    n_internal = n - 1
    fill!(store, zero(Int32))
    @metal threads=(ThreadsPerGroup,) groups=(cld(n, ThreadsPerGroup),) Partia.LinearBoundingVolumeHierarchy._initialize_leaf_node!(lbvh.scale, lbvh.aabb, scale, leaf_min, leaf_max, n_internal)

    if n_internal > 0
        # Same-queue command ordering removes the need for an intermediate
        # host-side synchronization between leaf initialization and ascent.
        @metal threads=(ThreadsPerGroup,) groups=(cld(n, ThreadsPerGroup),) Partia.LinearBoundingVolumeHierarchy._ascend_from_leaf!(lbvh, store, codes)
    end
    Metal.synchronize()
    return lbvh
end

function Partia.build!(lbvh :: LinearBVH{D, Float32, MtlVector{Float32}, MtlVector{Int32}}, store :: MtlVector{Int32}, enc :: MortonEncoding{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, scale :: MtlVector{Float32}, :: Val{ThreadsPerGroup} = Val(256)) where {D, ThreadsPerGroup, TI <: Unsigned}
    return Partia.build!(lbvh, store, enc, scale, enc.coord, enc.coord, Val(ThreadsPerGroup))
end

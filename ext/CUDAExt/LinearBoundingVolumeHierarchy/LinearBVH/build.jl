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
Morton-sorted leaves. The shorter form uses point AABBs from `enc.coord`.

# Returns
- `lbvh`: The rebuilt hierarchy after CUDA construction has completed.
"""
function Partia.build!(lbvh :: LinearBVH{D, TF, CuVector{TF}, CuVector{Int32}}, store :: CuVector{Int32}, enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, scale :: CuVector{TF}, leaf_min :: NTuple{D, CuVector{TF}}, leaf_max :: NTuple{D, CuVector{TF}}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned}
    codes = enc.codes
    n = length(codes)
    n == lbvh.nleaf || throw(DimensionMismatch("lbvh.nleaf and enc.codes must have identical lengths"))
    length(store) == n - 1 || throw(DimensionMismatch("store must have length lbvh.nleaf - 1"))
    length(scale) == n || throw(DimensionMismatch("scale and enc.codes must have identical lengths"))
    all(length(v) == n for v in leaf_min) || throw(DimensionMismatch("leaf_min and enc.codes must have identical lengths"))
    all(length(v) == n for v in leaf_max) || throw(DimensionMismatch("leaf_max and enc.codes must have identical lengths"))

    n_internal = n - 1
    fill!(store, zero(Int32))
    @cuda threads=ThreadsPerBlock blocks=NBlocks Partia.LinearBoundingVolumeHierarchy._initialize_leaf_node!(lbvh.scale, lbvh.aabb, scale, leaf_min, leaf_max, n_internal)

    if n_internal > 0
        # Same-stream launch ordering makes the initialized leaves visible to
        # the bottom-up topology construction without an intermediate barrier.
        @cuda threads=ThreadsPerBlock blocks=NBlocks Partia.LinearBoundingVolumeHierarchy._ascend_from_leaf!(lbvh, store, codes)
    end
    CUDA.synchronize()
    return lbvh
end

function Partia.build!(lbvh :: LinearBVH{D, TF, CuVector{TF}, CuVector{Int32}}, store :: CuVector{Int32}, enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, scale :: CuVector{TF}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned}
    return Partia.build!(lbvh, store, enc, scale, enc.coord, enc.coord, Val(NBlocks), Val(ThreadsPerBlock))
end

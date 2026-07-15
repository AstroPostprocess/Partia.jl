"""
    update!(lbvh, enc, scale, ::Val{ThreadsPerGroup}=Val(256))

Resize and rebuild a Metal-backed `LinearBVH` from a Morton-sorted encoding.
Topology, AABB, and hierarchical-scale device vectors are resized in place.
A temporary rendezvous store is allocated for the build.

# Parameters
- `lbvh`: Reusable Metal hierarchy storage.
- `enc`: Morton-sorted Metal encoding defining the leaf order and count.
- `scale`: Per-leaf scales in the same order as `enc`.
- `ThreadsPerGroup`: Number of threads per Metal threadgroup.

# Returns
- `LinearBVH`: The same `lbvh` wrapper after resizing and rebuilding.
"""
function Partia.update!(lbvh :: LinearBVH{D, Float32, MtlVector{Float32}, MtlVector{Int32}}, enc :: MortonEncoding{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, scale :: MtlVector{Float32}, :: Val{ThreadsPerGroup} = Val(256)) where {D, ThreadsPerGroup, TI <: Unsigned}
    n = length(enc.codes)
    n > 0 || throw(ArgumentError("LinearBVH requires at least one leaf"))
    length(scale) == n || throw(DimensionMismatch("scale and enc.codes must have identical lengths"))

    n_internal = n - 1
    total_length = 2n - 1
    resize!(lbvh.left, n_internal)
    resize!(lbvh.escape, total_length)
    resize!(lbvh.scale, total_length)
    @inbounds for d in 1:D
        resize!(lbvh.aabb.min[d], total_length)
        resize!(lbvh.aabb.max[d], total_length)
    end

    store = MtlVector{Int32}(undef, n_internal)
    Partia.build!(lbvh, store, enc, scale, Val(ThreadsPerGroup))
    return lbvh
end

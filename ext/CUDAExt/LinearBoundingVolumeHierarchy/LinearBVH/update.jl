"""
    update!(lbvh, enc, scale,
            ::Val{NBlocks}=Val(256),
            ::Val{ThreadsPerBlock}=Val(256))

Resize and rebuild a CUDA-backed `LinearBVH` from a Morton-sorted encoding.
Topology, AABB, and hierarchical-scale device vectors are resized in place.
A temporary rendezvous store is allocated for the build.

# Parameters
- `lbvh`: Reusable CUDA hierarchy storage.
- `enc`: Morton-sorted CUDA encoding defining the leaf order and count.
- `scale`: Per-leaf scales in the same order as `enc`.
- `NBlocks`: Number of CUDA blocks.
- `ThreadsPerBlock`: Number of threads per CUDA block.

# Returns
- `LinearBVH`: The same `lbvh` wrapper after resizing and rebuilding.
"""
function Partia.update!(lbvh :: LinearBVH{D, TF, CuVector{TF}, CuVector{Int32}}, enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, scale :: CuVector{TF}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned}
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

    store = CuVector{Int32}(undef, n_internal)
    Partia.build!(lbvh, store, enc, scale, Val(NBlocks), Val(ThreadsPerBlock))
    return lbvh
end

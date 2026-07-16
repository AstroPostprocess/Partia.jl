"""
    update!(lbvh, enc, scale)

Resize and rebuild a CPU-backed `LinearBVH` from a Morton-sorted encoding.
Topology, AABB, and hierarchical-scale vectors are resized in place. A
temporary rendezvous store is allocated for the build, and point coordinates
from `enc` are used as both leaf AABB endpoints.

# Parameters
- `lbvh`: Reusable CPU hierarchy storage.
- `enc`: Morton-sorted encoding that defines the new leaf order and count.
- `scale`: Per-leaf scale values in the same order as `enc`.

# Returns
- `LinearBVH`: The same `lbvh` wrapper after resizing and rebuilding.
"""
function update!(lbvh :: LinearBVH{D, TF, Vector{TF}, Vector{Int32}}, enc :: MortonEncoding{D, TF, TI, Vector{TF}, Vector{TI}}, scale :: Vector{TF}) where {D, TF <: AbstractFloat, TI <: Unsigned}
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

    store = Vector{Int32}(undef, n_internal)
    build!(lbvh, store, enc, scale)
    return lbvh
end

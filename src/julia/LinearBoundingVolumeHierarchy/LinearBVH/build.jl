######################################################################################

# Reusable linear bounding volume hierarchy builders.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
"""
    build!(lbvh, store, enc, scale, leaf_min, leaf_max)
    build!(lbvh, store, enc, scale)

Rebuild a preallocated `LinearBVH` from Morton-sorted leaves. The explicit form
uses the supplied leaf bounds; the shorter form treats `enc.coord` as point
AABBs. Both `lbvh` and the rendezvous `store` are reused without allocating
hierarchy storage.

# Parameters
- `lbvh`: Preallocated `Vector`-backed hierarchy with
  `nleaf == length(enc.codes)`.
- `store`: Reusable `Vector{Int32}` rendezvous storage of length `nleaf - 1`.
- `enc`: Morton encoding whose codes and coordinates have already been sorted
  by `sort_by_morton!`.
- `scale`: Per-leaf scale values arranged in the same Morton order as `enc`.
- `leaf_min`, `leaf_max`: Per-axis leaf bounds arranged in that same order.

# Returns
- `nothing`: `lbvh` and `store` are updated in place.
"""
function build!(lbvh :: LinearBVH{D, TF, Vector{TF}, Vector{Int32}}, store :: Vector{Int32}, enc :: MortonEncoding{D, TF, TI, Vector{TF}, Vector{TI}}, scale :: Vector{TF}, leaf_min :: NTuple{D, Vector{TF}}, leaf_max :: NTuple{D, Vector{TF}}) where {D, TF <: AbstractFloat, TI <: Unsigned}
    codes = enc.codes
    n = length(codes)
    n == lbvh.nleaf || throw(DimensionMismatch("lbvh.nleaf and enc.codes must have identical lengths"))
    length(store) == n - 1 || throw(DimensionMismatch("store must have length lbvh.nleaf - 1"))
    length(scale) == n || throw(DimensionMismatch("scale and enc.codes must have identical lengths"))
    @inbounds for d in 1:D
        length(leaf_min[d]) == n || throw(DimensionMismatch("leaf_min[$d] and enc.codes must have identical lengths"))
        length(leaf_max[d]) == n || throw(DimensionMismatch("leaf_max[$d] and enc.codes must have identical lengths"))
    end
    _issorted(codes) || throw(ArgumentError("LinearBVH: enc.codes must be sorted in nondecreasing order."))

    n_internal = n - 1
    fill!(store, zero(Int32))
    lbvh.escape[1] = zero(Int32)

    # Initialise the unified leaf section before constructing parent nodes.
    @threads for i in 1:n
        _initialize_leaf_node!(lbvh.scale, lbvh.aabb, scale, leaf_min, leaf_max, n_internal, i)
    end

    if n_internal > 0
        # The second child reaching a split merges the completed subtrees and
        # continues upward; store is reusable rendezvous storage.
        @threads for i in 1:n
            _ascend_from_leaf!(lbvh, store, codes, i)
        end
    end
    return nothing
end

function build!(lbvh :: LinearBVH{D, TF, Vector{TF}, Vector{Int32}}, store :: Vector{Int32}, enc :: MortonEncoding{D, TF, TI, Vector{TF}, Vector{TI}}, scale :: Vector{TF}) where {D, TF <: AbstractFloat, TI <: Unsigned}
    build!(lbvh, store, enc, scale, enc.coord, enc.coord)
    return nothing
end

######################################################################################

# Linear bounding volume hierarchy data structures and constructors.
#     by Wei-Shan Su,
#     May 4, 2026

######################################################################################
struct LinearBVH{D, TF <: AbstractFloat, VF <: AbstractVector{TF}, VB <: AbstractVector{Int32}}
    brt :: BinaryRadixTree{VB}
    leaf_coor :: NTuple{D, VF}
    leaf_scale :: VF
    node_aabb :: AABB{D, TF, VF}
    node_scale :: VF
end

function Adapt.adapt_structure(to, x :: LBVH) where {D, LBVH <: LinearBVH{D}}
    LinearBVH(
        Adapt.adapt(to, x.brt),
        ntuple(i -> Adapt.adapt(to, x.leaf_coor[i]), D),
        Adapt.adapt(to, x.leaf_scale),
        Adapt.adapt(to, x.node_aabb),
        Adapt.adapt(to, x.node_scale),
    )
end

################# Constructing LBVH #################
"""
        LinearBVH(enc :: MortonEncoding, brt :: BinaryRadixTree, scale)
        LinearBVH(enc :: MortonEncoding, brt :: BinaryRadixTree, box_scale :: BoxScale)

Assemble a linear bounding volume hierarchy from a Morton-encoded particle set
and its matching binary radix tree. This stores per-leaf particle coordinates,
allocates per-node axis-aligned bounding boxes, discovers the tree root, and
precomputes the hierarchical extent data required for subsequent neighbor queries.

# Parameters
- `enc :: MortonEncoding`: Morton-sorted particle coordinates and permutation.
- `brt :: BinaryRadixTree`: Connectivity generated from the same `enc` instance.
- `scale`: Per-particle scale vector. By default this is interpreted in the
    original particle order and reordered with `enc.order`.
- `box_scale :: BoxScale`: Per-particle scale vector plus whether it is already
    arranged in Morton order.

# Returns
- `LinearBVH`: Immutable hierarchy storing the tree topology, leaf particle
    coordinates, and internal-node bounding volumes.
"""
function LinearBVH(
    enc :: MortonEncoding{D, TF, TI, VF, VI},
    brt :: BinaryRadixTree{VB},
    scale :: VF,
) where {D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}, VB <: AbstractVector{Int32}}
    return LinearBVH(enc, brt, BoxScale(scale, false))
end

function LinearBVH(
    enc :: MortonEncoding{D, TF, TI, VF, VI},
    brt :: BinaryRadixTree{VB},
    box_scale :: BoxScale{TF, VF},
) where {D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}, VB <: AbstractVector{Int32}}
    scale = box_scale.scale
    nleaf = brt.nleaf
    ninternal = nleaf - 1
    ntotal = 2 * nleaf - 1
    length(scale) >= nleaf || throw(DimensionMismatch("scale length $(length(scale)) is shorter than nleaf $nleaf"))

    vproto = enc.coord[1]

    leaf_coor = ntuple(_ -> similar(vproto, nleaf), D)
    leaf_scale = similar(scale, nleaf)
    node_aabb = AABB(ntuple(_ -> similar(vproto, ninternal), D),
                     ntuple(_ -> similar(vproto, ninternal), D))
    node_scale = similar(scale, ninternal)

    LBVH = LinearBVH{D, TF, VF, VB}(brt, leaf_coor, leaf_scale, node_aabb, node_scale)

    visited = AtomicMemory{UInt32}(undef, ninternal)                        # atomic visit counters for internal nodes (2nd arrival processes the node)
    @threads for i in eachindex(visited)
        @inbounds @atomic :sequentially_consistent visited[i] = zero(UInt32)
    end

    _build_leaf_coords!(LBVH, enc)
    _build_leaf_scale!(LBVH, enc, box_scale)

    @threads for startid in Int32(nleaf):Int32(ntotal)                      # Karras: Each thread starts from one leaf node and walks up the tree using parent pointers that we record during radix tree construction.
        _build_internal_aabb!(LBVH, visited, startid)                       # Well I prefer to use id in 2n-1 space rather than the index of leaf
    end

    @inbounds for i in eachindex(visited)                                   # sanity: every internal node must be combined twice
        @assert (@atomic :sequentially_consistent visited[i]) >= UInt32(2)
    end

    return LBVH
end

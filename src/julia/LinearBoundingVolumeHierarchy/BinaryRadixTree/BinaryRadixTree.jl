######################################################################################

# Binary radix tree data structure and constructor for Morton-sorted particles.
#     by Wei-Shan Su,
#     May 4, 2026

######################################################################################
################# Define structures #################
struct BinaryRadixTree{V <: AbstractVector{Int32}}
    root :: Int32
    nleaf :: Int
    left :: V                     # length = 2*nleaf-1
    right :: V                     # length = 2*nleaf-1
    escape :: V                     # length = 2*nleaf-1
    parent :: V                     # length = 2*nleaf-1
end

function Adapt.adapt_structure(to, x :: BinaryRadixTree)
    BinaryRadixTree(
        x.root,
        x.nleaf,
        Adapt.adapt(to, x.left),
        Adapt.adapt(to, x.right),
        Adapt.adapt(to, x.escape),
        Adapt.adapt(to, x.parent)
    )
end

################# Constructing Binary Radix Tree #################
"""
    BinaryRadixTree(enc :: MortonEncoding)

Construct a **Binary Radix Tree (BRT)** from a Morton-sorted code array `enc.codes`,
using the Karras (2012) range/split construction and a **stackless escape table**
(Prokopenko & Lebrun-Grandié, 2024) for traversal without parent-walking.

The constructor builds the tree connectivity in *linear time* from the sorted Morton
codes. Children are stored as **unified node IDs** in a single node-id space of size
`2n-1` (A1 layout), where `n = length(enc.codes)`:

- internal node IDs: `1:(n-1)`
- leaf node IDs: `n:(2n-1)`  (leaf index `k ∈ 1:n` maps to node ID `(n-1) + k`)
- `0` is a sentinel meaning “no node” / termination

In addition, `escape` is computed for **all unified node IDs** (both internal and leaf).
During stackless traversal, `escape[id]` gives the next node to visit after finishing
(or pruning) the subtree rooted at `id`. Escape targets are derived from the adjacent
longest-common-prefix-length array `adj`, with a `+∞` sentinel at `adj[n]`.

This implementation also stores a **unified parent array** `parent` (length `2n-1`,
`parent[root]=0`). It is not required for stackless traversal when `escape` is used,
but is useful for bottom-up refit (Karras-style atomic “second arrival” reduction).

# Parameters
- `enc :: MortonEncoding{D, TF, TI, VF, VI}`
  Container holding the sorted Morton codes (`enc.codes`).

# Returns
- `BinaryRadixTree{Vector{Int32}}`
  A struct containing:
  - `root`   — root unified node ID (`1` if `n ≥ 2`, otherwise `0`)
  - `nleaf`  — number of leaves `n`
  - `left`   — left child node ID for each node (length `2n-1`; meaningful for internal nodes)
  - `right`  — right child node ID for each node (length `2n-1`; meaningful for internal nodes)
  - `escape` — escape node ID for each node (length `2n-1`; defined for both internal and leaf nodes)
  - `parent` — parent internal ID for each unified node ID (length `2n-1`; `0` for the root)

# Notes
- `enc.codes` must be sorted; otherwise the constructed tree and escape links are invalid.
- For leaf nodes, `left` and `right` are unused (typically left as `0`).
- This implementation stores node IDs in `Int32`. Therefore the maximum supported leaf count is
  `n ≤ 1_073_741_824` (since `2n-1` must fit in signed 32-bit).

# Reference
- Karras, T. (2012). *Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees*.
    In *High Performance Graphics 2012* (pp. 33–37).
    DOI: [10.2312/EGGH/HPG12/033-037]
- Prokopenko, A., & Lebrun-Grandié, D. (2024). *Revising Apetrei's bounding volume hierarchy
    construction algorithm to allow stackless traversal*. Oak Ridge National Laboratory
    Technical Report. DOI: 10.2172/2301619
"""
function BinaryRadixTree(enc :: MortonEncoding{D, TF, TI, VF, VI}) where {D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}}
    # Properties of BRT
    codes = enc.codes
    n = length(codes)                    # Int
    n >= 1 || throw(ArgumentError("BinaryRadixTree: enc.codes must be non-empty (got n=0)."))
    root = (n >= 2) ? one(Int32) : zero(Int32)

    # Int32 node-id capacity: total_length = 2n-1 must fit in Int32
    n_max = (typemax(Int32) ÷ 2) + 1  # 1_073_741_824
    n <= n_max || throw(ArgumentError("BinaryRadixTree: n=$n exceeds the Int32 node-id capacity (requires 2n-1 ≤ typemax(Int32); n ≤ $n_max)."))

    n_internal = n - 1                   # Int
    total_length = 2*n - 1               # Int

    # Initializing arrays (length = 2n - 1)
    left   = zeros(Int32, total_length)
    right  = zeros(Int32, total_length)
    escape = zeros(Int32, total_length)
    parent = zeros(Int32, total_length)

    # range right (length = ninternal)
    range_hi = Vector{Int32}(undef, n_internal)

    if n_internal > 0
        # Loop: establishing the Karras BRT
        @threads for i in 1:n_internal
            _build_child!(left, right, range_hi, parent, codes, i)
        end

        # Adjacent longest-common-prefix lengths (Algorithm 2, Prokopenko & Lebrun-Grandié 2024)
        adj = Vector{Int32}(undef, n)
        @threads for i in 1:n-1
            _build_adjacent!(adj, codes, i)
        end
        adj[n] = typemax(Int32)   # +∞ sentinel for the last slot

        # Internal-node escapes (Algorithm 2)
        @threads for i in 1:n_internal
            _build_escape!(escape, adj, range_hi, n, i)
        end

        # Leaf escapes are derived from parent + sibling relationships
        for leaf_id in n:total_length
            _build_escape!(escape, parent, left, right, leaf_id)
        end
    else
        escape[1] = 0
        parent[1] = 0
    end

    return BinaryRadixTree{Vector{Int32}}(root, n, left, right, escape, parent)
end

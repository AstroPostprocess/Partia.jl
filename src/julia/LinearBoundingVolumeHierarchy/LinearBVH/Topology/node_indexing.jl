######################################################################################

# Unified node-ID helpers shared by LBVH construction and traversal.
#     by Wei-Shan Su,
#     July 13, 2026

######################################################################################
"""
    is_leaf_id(node::Int32, nleaf::Int)

Return whether unified node ID `node` identifies a leaf. Node ID zero is the
stackless traversal sentinel and is never a leaf.

# Parameters
- `node`: Unified node ID.
- `nleaf`: Number of leaves in the hierarchy.

# Returns
- `Bool`: `true` when `node` lies in the leaf section.
"""
@inline is_leaf_id(node :: Int32, nleaf :: Int) = (node != 0) & (node >= Int32(nleaf))

"""
    leaf_index(node::Int32, nleaf::Int)

Convert a unified leaf-node ID into its one-based leaf-array index.

# Parameters
- `node`: Unified leaf-node ID.
- `nleaf`: Number of leaves in the hierarchy.

# Returns
- `Int`: One-based index within the leaf section.
"""
@inline leaf_index(node :: Int32, nleaf :: Int) = Int(node) - (nleaf - 1)

"""
    internal_index(node::Int32)

Convert a unified internal-node ID into its one-based internal-array index.

# Parameters
- `node`: Unified internal-node ID.

# Returns
- `Int`: One-based internal-array index.
"""
@inline internal_index(node :: Int32) = Int(node)

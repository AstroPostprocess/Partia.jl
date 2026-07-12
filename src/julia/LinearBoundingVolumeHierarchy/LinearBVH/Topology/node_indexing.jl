######################################################################################

# Unified node-ID helpers shared by LBVH construction and traversal.
#     by Wei-Shan Su,
#     July 13, 2026

######################################################################################
@inline is_leaf_id(node :: Int32, nleaf :: Int) = (node != 0) & (node >= Int32(nleaf))
@inline leaf_index(node :: Int32, nleaf :: Int) = Int(node) - (nleaf - 1)
@inline internal_index(node :: Int32) = Int(node)

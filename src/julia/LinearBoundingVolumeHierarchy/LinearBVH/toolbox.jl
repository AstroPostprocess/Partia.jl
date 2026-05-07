######################################################################################

# Linear bounding volume hierarchy build helpers for Morton-ordered leaves,
# per-node scale aggregation, and internal-node AABB construction.
#     by Wei-Shan Su,
#     May 4, 2026

######################################################################################
function _build_leaf_coords!(LBVH :: LinearBVH{D}, enc :: MortonEncoding{D}) where {D}
    coords = enc.coord
    leaf = LBVH.leaf_coor
    @inbounds for d in 1:D
        copyto!(leaf[d], coords[d])
    end
    return nothing
end

function _build_leaf_scale!(
    LBVH :: LinearBVH,
    enc :: MortonEncoding,
    box_scale :: BoxScale,
)
    leaf_scale = LBVH.leaf_scale
    scale = box_scale.scale
    if box_scale.is_morton_sorted
        copyto!(leaf_scale, 1, scale, 1, length(leaf_scale))
    else
        order = enc.order
        @inbounds for i in eachindex(leaf_scale)
            leaf_scale[i] = scale[order[i]]
        end
    end
    return nothing
end

function _build_internal_aabb!(LBVH :: LinearBVH{D}, visited :: AtomicMemory{UInt32}, startid :: Int32) where {D}
    brt = LBVH.brt
    nleaf = brt.nleaf

    # unified parent: length = 2nleaf-1, parent[root]=0, parent[leaf/internal]=parent internal id
    parent = brt.parent

    # AABB buffers
    node_min  = LBVH.node_aabb.min
    node_max  = LBVH.node_aabb.max
    node_scale = LBVH.node_scale
    leaf       = LBVH.leaf_coor
    leaf_scale = LBVH.leaf_scale

    # BRT children (only meaningful for internal ids 1..ninternal)
    left  = brt.left
    right = brt.right

    # climb starts from parent(startid); startid is a leaf unified node id in nleaf..2nleaf-1
    p = @inbounds parent[Int(startid)]   # internal id (1..ninternal) or 0

    while p != 0
        pidx = internal_index(p)

        # Karras: combine on the second arrival (newval == 2)
        newval = @atomic :sequentially_consistent visited[pidx] += one(UInt32)

        if newval == UInt32(2)
            # second arrival: both children are ready => combine
            @inbounds begin
                l = left[pidx]
                r = right[pidx]

                # scale
                hl = is_leaf_id(l, nleaf) ? leaf_scale[leaf_index(l, nleaf)] : node_scale[internal_index(l)]
                hr = is_leaf_id(r, nleaf) ? leaf_scale[leaf_index(r, nleaf)] : node_scale[internal_index(r)]
                node_scale[pidx] = ifelse(hl > hr, hl, hr)

                # bounds
                for d in 1:D
                    lmin = is_leaf_id(l, nleaf) ? leaf[d][leaf_index(l, nleaf)] : node_min[d][internal_index(l)]
                    rmin = is_leaf_id(r, nleaf) ? leaf[d][leaf_index(r, nleaf)] : node_min[d][internal_index(r)]
                    lmax = is_leaf_id(l, nleaf) ? leaf[d][leaf_index(l, nleaf)] : node_max[d][internal_index(l)]
                    rmax = is_leaf_id(r, nleaf) ? leaf[d][leaf_index(r, nleaf)] : node_max[d][internal_index(r)]
                    node_min[d][pidx] = ifelse(lmin < rmin, lmin, rmin)
                    node_max[d][pidx] = ifelse(lmax > rmax, lmax, rmax)
                end
            end

            # climb to parent of this internal node id (internal ids are valid indices in parent[])
            p = @inbounds parent[pidx]
        else
            # first arrival
            break
        end
    end

    return nothing
end

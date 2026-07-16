@inline function _ascend_from_leaf!(lbvh :: LinearBVH{D, TF, Vector{TF}, Vector{Int32}}, store :: Vector{Int32}, codes :: Vector{TI}, i :: Int) where {D, TF <: AbstractFloat, TI <: Unsigned}
    # Basic properties of LBVH
    n = nleaf(lbvh)
    leaf_offset = n - 1

    # Get arrays
    left = lbvh.left
    escape = lbvh.escape
    aabb = lbvh.aabb
    unified_scale = lbvh.scale

    # The completed subtree initially contains only leaf i.
    range_left = i
    range_right = i

    # LCP values at the two boundaries of the current range.
    delta_left = i > 1 ? _longest_common_prefix_length(codes, i - 1, i) : -1
    delta_right = i < n ? _longest_common_prefix_length(codes, i, i + 1) : -1

    # Install the escape index of leaf i.
    leaf_idx = i + leaf_offset
    if i == n
        escape[leaf_idx] = zero(Int32)
    else
        next_pos = i + 1
        next_delta = next_pos < n ? _longest_common_prefix_length(codes, next_pos, next_pos + 1) : -1
        escape[leaf_idx] = Int32(delta_right > next_delta ? next_pos + leaf_offset : next_pos)
    end

    while true
        if delta_right > delta_left
            # The current subtree is the left child of its parent.
            split = range_right
            result = Atomix.@atomicreplace store[split] zero(Int32) => Int32(range_left)
            result.success && return nothing

            range_right = Int(result.old)
            delta_right = range_right < n ? _longest_common_prefix_length(codes, range_right, range_right + 1) : -1
        else
            # The current subtree is the right child of its parent.
            split = range_left - 1
            result = Atomix.@atomicreplace store[split] zero(Int32) => Int32(range_right)
            result.success && return nothing

            range_left = Int(result.old)
            delta_left = range_left > 1 ? _longest_common_prefix_length(codes, range_left - 1, range_left) : -1
        end

        # Convert the completed parent range to the Karras internal-node index.
        parent_idx = delta_right > delta_left ? range_right : range_left

        # Determine both children from the split position.
        left_idx = split == range_left ? split + leaf_offset : split
        right_pos = split + 1
        right_idx = right_pos == range_right ? right_pos + leaf_offset : right_pos

        # Install the left child.
        left[parent_idx] = Int32(left_idx)

        # Install the escape index of the newly completed internal node.
        if range_right == n
            escape[parent_idx] = zero(Int32)
        else
            next_pos = range_right + 1
            next_delta = next_pos < n ? _longest_common_prefix_length(codes, next_pos, next_pos + 1) : -1
            escape[parent_idx] = Int32(delta_right > next_delta ? next_pos + leaf_offset : next_pos)
        end

        # Merge the child scale values.
        unified_scale[parent_idx] = max(unified_scale[left_idx], unified_scale[right_idx])

        # Merge the child AABBs.
        @inbounds for d in 1:D
            aabb.min[d][parent_idx] = min(aabb.min[d][left_idx], aabb.min[d][right_idx])
            aabb.max[d][parent_idx] = max(aabb.max[d][left_idx], aabb.max[d][right_idx])
        end

        # Internal node 1 is always the root in the Karras ordering.
        parent_idx == 1 && return nothing
    end
    return nothing
end

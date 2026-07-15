######################################################################################

# Bottom-up CUDA LBVH topology and hierarchy construction kernel.

######################################################################################
@inline function Partia.LinearBoundingVolumeHierarchy._ascend_from_leaf!(lbvh :: Partia.LinearBVH{D, TF, <:CuDeviceVector{TF}, <:CuDeviceVector{Int32}}, store :: CuDeviceVector{Int32}, codes :: CuDeviceVector{TI}) where {D, TF <: AbstractFloat, TI <: Unsigned}
    # Global one-based thread index and grid stride.
    i = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int(gridDim().x * blockDim().x)
    n = lbvh.nleaf
    leaf_offset = n - 1

    # A single leaf is also the root and escapes directly out of the tree.
    if n == 1
        i == 1 && (lbvh.escape[1] = zero(Int32))
        return nothing
    end

    while i <= n
        # The completed subtree initially consists of leaf i alone.
        range_left = i
        range_right = i
        # LCP values decide whether this subtree is the left or right child of
        # its next parent in Karras ordering.
        delta_left = i > 1 ? Partia.LinearBoundingVolumeHierarchy._longest_common_prefix_length(codes, i - 1, i) : -1
        delta_right = i < n ? Partia.LinearBoundingVolumeHierarchy._longest_common_prefix_length(codes, i, i + 1) : -1
        leaf_idx = i + leaf_offset

        # Install the stackless escape link for the leaf.
        if i == n
            lbvh.escape[leaf_idx] = zero(Int32)
        else
            next_pos = i + 1
            next_delta = next_pos < n ? Partia.LinearBoundingVolumeHierarchy._longest_common_prefix_length(codes, next_pos, next_pos + 1) : -1
            lbvh.escape[leaf_idx] = Int32(delta_right > next_delta ? next_pos + leaf_offset : next_pos)
        end

        while true
            # Publish all leaf/parent data written by this subtree before its
            # boundary becomes visible to a thread in another CUDA block.
            CUDA.threadfence()

            if delta_right > delta_left
                # Current subtree is the left child. The first arriving child
                # stores its range boundary; the second receives it via CAS.
                split = range_right
                old = CUDA.atomic_cas!(pointer(store, split), zero(Int32), Int32(range_left))
                iszero(old) && break
                range_right = Int(old)
                delta_right = range_right < n ? Partia.LinearBoundingVolumeHierarchy._longest_common_prefix_length(codes, range_right, range_right + 1) : -1
            else
                # Current subtree is the right child at split range_left - 1.
                split = range_left - 1
                old = CUDA.atomic_cas!(pointer(store, split), zero(Int32), Int32(range_right))
                iszero(old) && break
                range_left = Int(old)
                delta_left = range_left > 1 ? Partia.LinearBoundingVolumeHierarchy._longest_common_prefix_length(codes, range_left - 1, range_left) : -1
            end

            # Both children are now complete. Convert their range and split to
            # the unified Karras node indices.
            parent_idx = delta_right > delta_left ? range_right : range_left
            left_idx = split == range_left ? split + leaf_offset : split
            right_pos = split + 1
            right_idx = right_pos == range_right ? right_pos + leaf_offset : right_pos
            lbvh.left[parent_idx] = Int32(left_idx)

            # Install the stackless escape link for the completed parent.
            if range_right == n
                lbvh.escape[parent_idx] = zero(Int32)
            else
                next_pos = range_right + 1
                next_delta = next_pos < n ? Partia.LinearBoundingVolumeHierarchy._longest_common_prefix_length(codes, next_pos, next_pos + 1) : -1
                lbvh.escape[parent_idx] = Int32(delta_right > next_delta ? next_pos + leaf_offset : next_pos)
            end

            # Merge hierarchical scale and bounding boxes from both children.
            lbvh.scale[parent_idx] = max(lbvh.scale[left_idx], lbvh.scale[right_idx])
            @inbounds for d in 1:D
                lbvh.aabb.min[d][parent_idx] = min(lbvh.aabb.min[d][left_idx], lbvh.aabb.min[d][right_idx])
                lbvh.aabb.max[d][parent_idx] = max(lbvh.aabb.max[d][left_idx], lbvh.aabb.max[d][right_idx])
            end
            # Internal node 1 is the root, so this ascent is complete.
            parent_idx == 1 && break
        end
        i += stride
    end
    return nothing
end

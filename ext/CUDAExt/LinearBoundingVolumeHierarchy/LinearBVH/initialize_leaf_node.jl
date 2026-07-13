######################################################################################

# CUDA kernel for initializing the unified LBVH leaf section.

######################################################################################
@inline function Partia.LinearBoundingVolumeHierarchy._initialize_leaf_node!(unified_scale :: CuDeviceVector{TF}, aabb :: AABB{D, TF, <:CuDeviceVector{TF}}, scale :: CuDeviceVector{TF}, leaf_min :: NTuple{D, CuDeviceVector{TF}}, leaf_max :: NTuple{D, CuDeviceVector{TF}}, leaf_offset :: Int) where {D, TF <: AbstractFloat}
    # Global one-based thread index and grid stride.
    i = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int(gridDim().x * blockDim().x)
    # Each input leaf i is stored at unified node i + leaf_offset.
    while i <= length(scale)
        leaf_idx = i + leaf_offset
        @inbounds begin
            unified_scale[leaf_idx] = scale[i]
            for d in 1:D
                aabb.min[d][leaf_idx] = leaf_min[d][i]
                aabb.max[d][leaf_idx] = leaf_max[d][i]
            end
        end
        i += stride
    end
    return nothing
end

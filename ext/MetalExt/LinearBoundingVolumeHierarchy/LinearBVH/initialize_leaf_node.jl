@inline function Partia.LinearBoundingVolumeHierarchy._initialize_leaf_node!(unified_scale :: MtlDeviceVector{Float32, 1}, aabb :: AABB{D, Float32, MtlDeviceVector{Float32, 1}}, scale :: MtlDeviceVector{Float32, 1}, leaf_min :: NTuple{D, MtlDeviceVector{Float32, 1}}, leaf_max :: NTuple{D, MtlDeviceVector{Float32, 1}}, leaf_offset :: Int) where {D}
    # Get the global thread index and stride
    tid = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)

    n = length(scale)
    i = tid

    while i <= n
        leaf_idx = i + leaf_offset

        @inbounds begin
            unified_scale[leaf_idx] = scale[i]
            for d in 1:D
                lmin = leaf_min[d][i]
                lmax = leaf_max[d][i]

                aabb.min[d][leaf_idx] = lmin
                aabb.max[d][leaf_idx] = lmax
            end
        end

        i += stride
    end
    return nothing
end

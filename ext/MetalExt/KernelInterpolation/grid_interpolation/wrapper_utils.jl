function Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input :: AbstractInterpolationInput{D, Float32, MtlVector{Float32}}, LBVH :: LinearBVH{D, Float32, MtlVector{Float32}}) where {D}
    input.Npart == nleaf(LBVH) || throw(ArgumentError(
        "Provided LBVH leaf count does not match the interpolation input."
    ))

    input.Npart == 0 && return nothing

    flag = Metal.zeros(Int32, 1)
    threads = 256
    groups = cld(input.Npart, threads)
    @metal threads=threads groups=groups _validate_interpolation_lbvh_leaf_order_kernel!(flag, input, LBVH)

    only(Array(flag)) == 0 || throw(ArgumentError(
        "Provided LBVH leaf order does not match the current input ordering. " *
        "Ensure the LBVH was built from the same Morton-reordered input."
    ))
    return nothing
end

@inline function _validate_interpolation_lbvh_leaf_order_kernel!(flag :: MtlDeviceVector{Int32, 1}, input :: AbstractInterpolationInput{D, Float32, VF}, LBVH :: LinearBVH) where {D, VF <: MtlDeviceVector{Float32}}
    i = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)
    leaf_offset = nleaf(LBVH) - 1
    ptr = pointer(flag)

    while i <= input.Npart
        leaf = leaf_offset + i
        matches = input.h[i] == LBVH.scale[leaf]
        @inbounds for d in 1:D
            matches &= input.coord[d][i] == LBVH.aabb.min[d][leaf]
        end
        matches || unsafe_store!(ptr, Int32(1))
        i += stride
    end
    return nothing
end

function Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input :: AbstractInterpolationInput{D, TF, CuVector{TF}}, LBVH :: LinearBVH{D, TF, CuVector{TF}}) where {D, TF <: AbstractFloat}
    input.Npart == nleaf(LBVH) || throw(ArgumentError(
        "Provided LBVH leaf count does not match the interpolation input."
    ))

    input.Npart == 0 && return nothing

    flag = CUDA.zeros(Int32, 1)
    threads = 256
    blocks = cld(input.Npart, threads)
    @cuda threads=threads blocks=blocks _validate_interpolation_lbvh_leaf_order_kernel!(flag, input, LBVH)

    only(Array(flag)) == 0 || throw(ArgumentError(
        "Provided LBVH leaf order does not match the current input ordering. " *
        "Ensure the LBVH was built from the same Morton-reordered input."
    ))
    return nothing
end

@inline function _validate_interpolation_lbvh_leaf_order_kernel!(flag :: CuDeviceVector{Int32}, input :: AbstractInterpolationInput{D, TF, VF}, LBVH :: LinearBVH) where {D, TF <: AbstractFloat, VF <: CuDeviceVector{TF}}
    i = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int(gridDim().x * blockDim().x)
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

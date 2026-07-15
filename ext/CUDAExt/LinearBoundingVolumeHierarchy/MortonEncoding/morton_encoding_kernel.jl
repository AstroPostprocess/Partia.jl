######################################################################################

# Fused coordinate normalization, quantization, and Morton encoding CUDA kernel.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
@inline function Partia.LinearBoundingVolumeHierarchy._morton_encoding_kernel!(codes :: CuDeviceVector{TI}, coords :: NTuple{D, CuDeviceVector{TF}}, inv_extent :: NTuple{D, TF}, offset :: NTuple{D, TF}) where {D, TF <: AbstractFloat, TI <: Unsigned}
    # Global one-based thread index and grid stride.
    i = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int(gridDim().x * blockDim().x)
    # Quantization scale is derived from dimensionality and Morton-code width.
    scale = Partia.LinearBoundingVolumeHierarchy._axis_scale(Val(D), TI, TF)
    lo = zero(TF)
    hi = one(TF)

    # Grid-stride loop permits a fixed block count for arbitrary particle counts.
    while i <= length(codes)
        if D == 2
            # Normalize, clamp against roundoff at the box boundary, quantize,
            # and interleave both axes without allocating intermediate arrays.
            ix = unsafe_trunc(TI, scale * clamp(muladd(coords[1][i], inv_extent[1], offset[1]), lo, hi))
            iy = unsafe_trunc(TI, scale * clamp(muladd(coords[2][i], inv_extent[2], offset[2]), lo, hi))
            @inbounds codes[i] = Partia.LinearBoundingVolumeHierarchy._encode_morton_code2D(ix, iy)
        else
            # The constructor restricts D to 2 or 3, so this is the 3D path.
            ix = unsafe_trunc(TI, scale * clamp(muladd(coords[1][i], inv_extent[1], offset[1]), lo, hi))
            iy = unsafe_trunc(TI, scale * clamp(muladd(coords[2][i], inv_extent[2], offset[2]), lo, hi))
            iz = unsafe_trunc(TI, scale * clamp(muladd(coords[3][i], inv_extent[3], offset[3]), lo, hi))
            @inbounds codes[i] = Partia.LinearBoundingVolumeHierarchy._encode_morton_code3D(ix, iy, iz)
        end
        i += stride
    end
    return nothing
end

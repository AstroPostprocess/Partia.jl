######################################################################################

# Fused coordinate quantization and Morton encoding kernels.
#     by Wei-Shan Su,
#     July 13, 2026

######################################################################################

@inline function Partia.LinearBoundingVolumeHierarchy._morton_encoding_kernel!(codes :: MtlDeviceVector{TI}, coords :: NTuple{2, MtlDeviceVector{Float32}}, invΔ :: NTuple{2, Float32}, c :: NTuple{2, Float32}) where {TI <: Unsigned}
    # Get the global thread index and stride
    tid = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)

    n = length(codes)
    i = tid

    # Unpack the coordinate normalization parameters
    x, y = coords
    invΔx, invΔy = invΔ
    cx, cy = c

    # Compute the coordinate quantization scale
    scale = Partia.LinearBoundingVolumeHierarchy._axis_scale(Val(2), TI, Float32)

    # Encode all points assigned to this thread
    while i <= n
        @inbounds begin
            xi = x[i]
            yi = y[i]
        end

        fxi = clamp(muladd(xi, invΔx, cx), 0.0f0, 1.0f0)
        fyi = clamp(muladd(yi, invΔy, cy), 0.0f0, 1.0f0)

        ixi = unsafe_trunc(TI, scale * fxi)
        iyi = unsafe_trunc(TI, scale * fyi)

        @inbounds codes[i] = Partia.LinearBoundingVolumeHierarchy._encode_morton_code2D(ixi, iyi)

        i += stride
    end
    return nothing
end

@inline function Partia.LinearBoundingVolumeHierarchy._morton_encoding_kernel!(codes :: MtlDeviceVector{TI}, coords :: NTuple{3, MtlDeviceVector{Float32}}, invΔ :: NTuple{3, Float32}, c :: NTuple{3, Float32}) where {TI <: Unsigned}
    # Get the global thread index and stride
    tid = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)

    n = length(codes)
    i = tid

    # Unpack the coordinate normalization parameters
    x, y, z = coords
    invΔx, invΔy, invΔz = invΔ
    cx, cy, cz = c

    # Compute the coordinate quantization scale
    scale = Partia.LinearBoundingVolumeHierarchy._axis_scale(Val(3), TI, Float32)

    # Encode all points assigned to this thread
    while i <= n
        @inbounds begin
            xi = x[i]
            yi = y[i]
            zi = z[i]
        end

        fxi = clamp(muladd(xi, invΔx, cx), 0.0f0, 1.0f0)
        fyi = clamp(muladd(yi, invΔy, cy), 0.0f0, 1.0f0)
        fzi = clamp(muladd(zi, invΔz, cz), 0.0f0, 1.0f0)

        ixi = unsafe_trunc(TI, scale * fxi)
        iyi = unsafe_trunc(TI, scale * fyi)
        izi = unsafe_trunc(TI, scale * fzi)

        @inbounds codes[i] = Partia.LinearBoundingVolumeHierarchy._encode_morton_code3D(ixi, iyi, izi)

        i += stride
    end
    return nothing
end

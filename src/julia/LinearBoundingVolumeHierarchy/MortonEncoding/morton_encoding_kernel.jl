######################################################################################

# Fused coordinate quantization and Morton encoding kernels.
#     by Wei-Shan Su,
#     July 12, 2026

######################################################################################

@inline function _morton_encoding_kernel!(codes :: Vector{TI}, i :: Int, coords :: NTuple{2, Vector{T}}, invΔ :: NTuple{2, T}, c :: NTuple{2, T}) where {TI <: Unsigned, T <: AbstractFloat}
    x, y = coords
    invΔx, invΔy = invΔ
    cx, cy = c

    scale = _axis_scale(Val(2), TI, T)

    @inbounds begin
        xi = x[i]
        yi = y[i]
    end

    fxi = clamp(muladd(xi, invΔx, cx), zero(T), one(T))
    fyi = clamp(muladd(yi, invΔy, cy), zero(T), one(T))

    ixi = TI(floor(scale * fxi))
    iyi = TI(floor(scale * fyi))

    @inbounds codes[i] = _encode_morton_code2D(ixi, iyi)
    return nothing
end

@inline function _morton_encoding_kernel!(codes :: Vector{TI}, i :: Int, coords :: NTuple{3, Vector{T}}, invΔ :: NTuple{3, T}, c :: NTuple{3, T}) where {TI <: Unsigned, T <: AbstractFloat}
    x, y, z = coords
    invΔx, invΔy, invΔz = invΔ
    cx, cy, cz = c

    scale = _axis_scale(Val(3), TI, T)

    @inbounds begin
        xi = x[i]
        yi = y[i]
        zi = z[i]
    end

    fxi = clamp(muladd(xi, invΔx, cx), zero(T), one(T))
    fyi = clamp(muladd(yi, invΔy, cy), zero(T), one(T))
    fzi = clamp(muladd(zi, invΔz, cz), zero(T), one(T))

    ixi = TI(floor(scale * fxi))
    iyi = TI(floor(scale * fyi))
    izi = TI(floor(scale * fzi))

    @inbounds codes[i] = _encode_morton_code3D(ixi, iyi, izi)
    return nothing
end

######################################################################################

# # Density

######################################################################################
@inline function _density_accumulation(Δr :: T, mb :: T, h :: T, smoothed_kernel :: K, :: Val{D} = Val(3)) where {T <: AbstractFloat, K <: AbstractSPHKernel, D}
    Ktyp = typeof(smoothed_kernel)
    W = Smoothed_kernel_function(Ktyp, Δr, h, Val(D))
    return mb * W
end

@inline function _density_accumulation(ra :: NTuple{D, T}, rb :: NTuple{D, T}, mb :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel, D}
    Ktyp = typeof(smoothed_kernel)
    W = Smoothed_kernel_function(Ktyp, ra, rb, h)
    return mb * W
end

## Number density
@inline function _number_density_accumulation(Δr :: T, h :: T, smoothed_kernel :: K, :: Val{D} = Val(3)) where {T <: AbstractFloat, K <: AbstractSPHKernel, D}
    Ktyp = typeof(smoothed_kernel)
    W = Smoothed_kernel_function(Ktyp, Δr, h, Val(D))
    return W
end

@inline function _number_density_accumulation(ra :: NTuple{D, T}, rb :: NTuple{D, T}, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel, D}
    Ktyp = typeof(smoothed_kernel)
    W = Smoothed_kernel_function(Ktyp, ra, rb, h)
    return W
end

## Single quantity intepolation
@inline function _quantity_interpolate_accumulation(Δr :: T, mb :: T, ρb :: T, Ab :: T, h :: T, smoothed_kernel :: K, :: Val{D} = Val(3)) where {T <: AbstractFloat, K <: AbstractSPHKernel, D}
    Ktyp = typeof(smoothed_kernel)
    W = Smoothed_kernel_function(Ktyp, Δr, h, Val(D))
    mbWlρb = mb * W/ρb
    return Ab * mbWlρb
end

@inline function _quantity_interpolate_accumulation(ra :: NTuple{D, T}, rb :: NTuple{D, T}, mb :: T, ρb :: T, Ab :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel, D}
    Ktyp = typeof(smoothed_kernel)
    W = Smoothed_kernel_function(Ktyp, ra, rb, h)
    mbWlρb = mb * W/ρb
    return Ab * mbWlρb
end

# Shepard Normalization
@inline function _ShepardNormalization_accumulation(Δr :: T, mb :: T, ρb :: T, h :: T, smoothed_kernel :: K, :: Val{D} = Val(3)) where {T <: AbstractFloat, K <: AbstractSPHKernel, D}
    Ktyp = typeof(smoothed_kernel)
    W = Smoothed_kernel_function(Ktyp, Δr, h, Val(D))
    mbWlρb = mb * W/ρb
    return mbWlρb
end

@inline function _ShepardNormalization_accumulation(ra :: NTuple{D, T}, rb :: NTuple{D, T}, mb :: T, ρb :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel, D}
    Ktyp = typeof(smoothed_kernel)
    W = Smoothed_kernel_function(Ktyp, ra, rb, h)
    mbWlρb = mb * W/ρb
    return mbWlρb
end

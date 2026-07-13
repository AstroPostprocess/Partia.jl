######################################################################################

# Structured-grid coordinate expansion kernels for CUDA.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################

# Expand Cartesian axes directly into Cartesian coordinates.

######################################################################################
@inline function _cartesian_coordinate_grid_kernel!(coor :: NTuple{D, CuDeviceVector{TF}}, axes :: NTuple{D, CuDeviceVector{TF}}, sz :: NTuple{D, Int}) where {D, TF <: AbstractFloat}
    i = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int(gridDim().x * blockDim().x)

    while i <= length(coor[1])
        # Convert column-major linear indexing into one index per axis.
        remainder = i - 1
        @inbounds for d in eachindex(coor)
            index = rem(remainder, sz[d]) + 1
            remainder = div(remainder, sz[d])
            coor[d][i] = axes[d][index]
        end
        i += stride
    end
    return nothing
end

######################################################################################

# Expand polar axes and convert every sample to Cartesian coordinates.

######################################################################################
@inline function _polar_coordinate_grid_kernel!(coor :: NTuple{2, CuDeviceVector{TF}}, axes :: NTuple{2, CuDeviceVector{TF}}, sz :: NTuple{2, Int}) where {TF <: AbstractFloat}
    i = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int(gridDim().x * blockDim().x)

    while i <= length(coor[1])
        q = i - 1
        i1 = rem(q, sz[1]) + 1
        i2 = rem(div(q, sz[1]), sz[2]) + 1
        @inbounds coor[1][i], coor[2][i] = Partia.Tools._cylin2cart(axes[1][i1], axes[2][i2])
        i += stride
    end
    return nothing
end

######################################################################################

# Expand cylindrical axes and convert every sample to Cartesian coordinates.

######################################################################################
@inline function _cylindrical_coordinate_grid_kernel!(coor :: NTuple{3, CuDeviceVector{TF}}, axes :: NTuple{3, CuDeviceVector{TF}}, sz :: NTuple{3, Int}) where {TF <: AbstractFloat}
    i = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int(gridDim().x * blockDim().x)

    while i <= length(coor[1])
        q = i - 1
        i1 = rem(q, sz[1]) + 1
        q = div(q, sz[1])
        i2 = rem(q, sz[2]) + 1
        i3 = rem(div(q, sz[2]), sz[3]) + 1
        @inbounds coor[1][i], coor[2][i], coor[3][i] = Partia.Tools._cylin2cart(axes[1][i1], axes[2][i2], axes[3][i3])
        i += stride
    end
    return nothing
end

######################################################################################

# Expand spherical axes and convert every sample to Cartesian coordinates.

######################################################################################
@inline function _spherical_coordinate_grid_kernel!(coor :: NTuple{3, CuDeviceVector{TF}}, axes :: NTuple{3, CuDeviceVector{TF}}, sz :: NTuple{3, Int}) where {TF <: AbstractFloat}
    i = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int(gridDim().x * blockDim().x)

    while i <= length(coor[1])
        q = i - 1
        i1 = rem(q, sz[1]) + 1
        q = div(q, sz[1])
        i2 = rem(q, sz[2]) + 1
        i3 = rem(div(q, sz[2]), sz[3]) + 1
        @inbounds coor[1][i], coor[2][i], coor[3][i] = Partia.Tools._sph2cart(axes[1][i1], axes[2][i2], axes[3][i3])
        i += stride
    end
    return nothing
end

abstract type AbstractInterpolationInput{D, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel, NCOLUMN} end


# Some useful function
## Get "Valid" range of data (the other would be 0)
@inline Base.length(input :: AbstractInterpolationInput)= input.Npart

## Get element type of the input
@inline Base.eltype( :: AbstractInterpolationInput{D, T}) where {D, T <: AbstractFloat} = T

## Get dimension of the input
@inline spatial_dimension( :: AbstractInterpolationInput{D}) where {D} = D

## Coordinate accessors
@inline get_coord(input :: AbstractInterpolationInput{D}) where {D} = input.coord
@inline get_xcoord(input :: AbstractInterpolationInput{D}) where {D} = input.coord[1]
@inline get_ycoord(input :: AbstractInterpolationInput{D}) where {D} = input.coord[2]
@inline get_zcoord(input :: AbstractInterpolationInput{3}) = input.coord[3]
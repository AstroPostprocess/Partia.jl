"""
    AbstractInterpolationInput{D, T, V, K, NCOLUMN}

Abstract supertype for particle-side interpolation inputs in `D` dimensions.
`T` is the floating-point type, `V` the vector storage type, `K` the SPH
kernel type, and `NCOLUMN` the number of carried quantity columns.
"""
abstract type AbstractInterpolationInput{D, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel, NCOLUMN} end


# Some useful function
## Get "Valid" range of data (the other would be 0)
@inline Base.length(input :: AbstractInterpolationInput)= input.Npart

## Get element type of the input
@inline Base.eltype( :: AbstractInterpolationInput{D, T}) where {D, T <: AbstractFloat} = T

## Get dimension of the input
"""
    spatial_dimension(input::AbstractInterpolationInput{D})

Return the compile-time spatial dimension `D` of an interpolation input.

# Parameters
- `input`: Particle-side interpolation input.

# Returns
- `Int`: Spatial dimension.
"""
@inline spatial_dimension( :: AbstractInterpolationInput{D}) where {D} = D

## Coordinate accessors
"""
    get_coord(input::AbstractInterpolationInput)

Return the complete structure-of-arrays coordinate tuple stored in `input`.

# Parameters
- `input`: Particle-side interpolation input.

# Returns
- `Tuple`: Per-axis coordinate vectors.
"""
@inline get_coord(input :: AbstractInterpolationInput{D}) where {D} = input.coord

"""
    get_xcoord(input::AbstractInterpolationInput)

Return the x-coordinate vector stored in an interpolation input.

# Parameters
- `input`: Particle-side interpolation input.

# Returns
- `AbstractVector`: Stored x coordinates.
"""
@inline get_xcoord(input :: AbstractInterpolationInput{D}) where {D} = input.coord[1]

"""
    get_ycoord(input::AbstractInterpolationInput)

Return the y-coordinate vector stored in an interpolation input.

# Parameters
- `input`: Particle-side interpolation input.

# Returns
- `AbstractVector`: Stored y coordinates.
"""
@inline get_ycoord(input :: AbstractInterpolationInput{D}) where {D} = input.coord[2]

"""
    get_zcoord(input::AbstractInterpolationInput{3})

Return the z-coordinate vector stored in a three-dimensional interpolation
input.

# Parameters
- `input`: Three-dimensional particle-side interpolation input.

# Returns
- `AbstractVector`: Stored z coordinates.
"""
@inline get_zcoord(input :: AbstractInterpolationInput{3}) = input.coord[3]

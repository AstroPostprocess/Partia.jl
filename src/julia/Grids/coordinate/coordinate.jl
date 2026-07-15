######################################################################################

# Dispatch tag for constructing grid

######################################################################################
"""
    AxisParam{TF} = Tuple{TF, TF, Int}

Axis specification tuple `(xmin, xmax, xn)`.

# Type Parameters
- `TF <: AbstractFloat` : Floating-point type for axis endpoints.

# Fields / Layout
- `xmin :: TF` : Axis minimum.
- `xmax :: TF` : Axis maximum.
- `xn :: Int`  : Number of points.
"""
const AxisParam{TF} = Tuple{TF, TF, Int}

"""Abstract supertype for coordinate-system dispatch tags."""
abstract type AbstractCoordinateSystem end

"""Cartesian coordinate-system dispatch tag."""
struct Cartesian <: AbstractCoordinateSystem end

"""Two-dimensional polar coordinate-system dispatch tag `(s, ?)`."""
struct Polar <: AbstractCoordinateSystem end        # (s, ?)

"""Three-dimensional cylindrical coordinate-system dispatch tag `(s, ?, z)`."""
struct Cylindrical <: AbstractCoordinateSystem end        # (s, ?, z)

"""Three-dimensional spherical coordinate-system dispatch tag `(r, ?, 庛)`."""
struct Spherical <: AbstractCoordinateSystem end        # (r, ?, 庛)


@inline function _coordinate_grid_isapprox(
    actual :: NTuple{D, VA},
    expected :: NTuple{D, VE};
    atol :: Real = 1.0e-8,
    rtol :: Real = 1.0e-8,
) where {D, T <: AbstractFloat, VA <: AbstractVector{T}, VE <: AbstractVector{T}}
    @inbounds for d in 1:D
        length(actual[d]) == length(expected[d]) || return false
        for i in eachindex(actual[d], expected[d])
            isapprox(actual[d][i], expected[d][i]; atol = atol, rtol = rtol) || return false
        end
    end
    return true
end


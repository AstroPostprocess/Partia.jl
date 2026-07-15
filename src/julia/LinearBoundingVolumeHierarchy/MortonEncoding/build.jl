######################################################################################

# Reusable and no-copy Morton encoding builders.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
"""
    build!(enc, points)
    build!(enc, x, y)
    build!(enc, x, y, z)

Recompute a preallocated Morton encoding from unsorted structure-of-arrays
coordinates. Code and coordinate storage are reused, and input coordinates are
copied into `enc.coord` before their Morton codes are recomputed. This function
does not sort the result; call `sort_by_morton!`
before constructing a `LinearBVH`.

# Parameters
- `enc`: Preallocated two- or three-dimensional `MortonEncoding` to rebuild.
- `points`: Unsorted coordinates in structure-of-arrays form.
- `x`, `y`, `z`: Coordinate-wise convenience arguments for `points`.

# Returns
- `nothing`: `enc.codes` and `enc.coord` are updated in place in input order.
"""
function build!(enc :: MortonEncoding{D, TF, TI, Vector{TF}, Vector{TI}}, points :: NTuple{D, Vector{TF}}) where {D, TF <: AbstractFloat, TI <: Unsigned}
    D in (2, 3) || throw(ArgumentError("Morton encoding only supports two or three dimensions"))
    n = length(enc.codes)
    n > 0 || throw(ArgumentError("coordinates must not be empty"))
    length(enc.order) == n || throw(DimensionMismatch("enc.order and enc.codes must have identical lengths"))
    all(length(p) == n for p in points) || throw(DimensionMismatch("points and enc.codes must have identical lengths"))
    all(axes(p) == axes(points[1]) for p in points) || throw(DimensionMismatch("coordinates must have identical axes"))

    # Restore unsorted coordinates into reusable encoding storage.
    for d in 1:D
        copyto!(enc.coord[d], points[d])
    end

    bounds = _coordinate_bounds(points)
    inv_extent = ntuple(D) do d
        extent = bounds[d][2] - bounds[d][1]
        iszero(extent) ? zero(TF) : inv(extent)
    end
    offset = ntuple(D) do d
        iszero(inv_extent[d]) ? TF(0.5) : -inv_extent[d] * bounds[d][1]
    end

    @inbounds @threads for i in eachindex(enc.codes)
        _morton_encoding_kernel!(enc.codes, i, enc.coord, inv_extent, offset)
    end
    return nothing
end

function build!(enc :: MortonEncoding{2, TF, TI, Vector{TF}, Vector{TI}}, x :: Vector{TF}, y :: Vector{TF}) where {TF <: AbstractFloat, TI <: Unsigned}
    build!(enc, (x, y))
    return nothing
end

function build!(enc :: MortonEncoding{3, TF, TI, Vector{TF}, Vector{TI}}, x :: Vector{TF}, y :: Vector{TF}, z :: Vector{TF}) where {TF <: AbstractFloat, TI <: Unsigned}
    build!(enc, (x, y, z))
    return nothing
end

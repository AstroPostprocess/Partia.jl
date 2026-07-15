"""
    update!(enc, coords)
    update!(enc, x, y)
    update!(enc, x, y, z)

Resize and rebuild a CPU-backed `MortonEncoding` from new unsorted coordinates.
The existing `order`, `codes`, and coordinate vectors are resized in place,
after which `build!` copies the supplied coordinates into `enc.coord` and
recomputes the Morton codes. The result remains unsorted.

# Parameters
- `enc`: Reusable two- or three-dimensional CPU Morton encoding.
- `coords`: Coordinate vectors in structure-of-arrays form.
- `x`, `y`, `z`: Coordinate-wise convenience arguments.

# Returns
- `MortonEncoding`: The same `enc` wrapper with resized and updated storage.
"""
function update!(enc :: MortonEncoding{D, TF, TI, Vector{TF}, Vector{TI}}, coords :: NTuple{D, Vector{TF}}) where {D, TF <: AbstractFloat, TI <: Unsigned}
    D in (2, 3) || throw(ArgumentError("Morton encoding only supports two or three dimensions"))
    n = length(coords[1])
    n > 0 || throw(ArgumentError("coordinates must not be empty"))
    all(length(coord) == n for coord in coords) || throw(DimensionMismatch("coordinates must have identical lengths"))

    resize!(enc.order, n)
    resize!(enc.codes, n)
    @inbounds for d in 1:D
        resize!(enc.coord[d], n)
    end

    build!(enc, coords)
    return enc
end

function update!(enc :: MortonEncoding{2, TF, TI, Vector{TF}, Vector{TI}}, x :: Vector{TF}, y :: Vector{TF}) where {TF <: AbstractFloat, TI <: Unsigned}
    return update!(enc, (x, y))
end

function update!(enc :: MortonEncoding{3, TF, TI, Vector{TF}, Vector{TI}}, x :: Vector{TF}, y :: Vector{TF}, z :: Vector{TF}) where {TF <: AbstractFloat, TI <: Unsigned}
    return update!(enc, (x, y, z))
end

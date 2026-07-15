"""
    update!(enc, coords, ::Val{ThreadsPerGroup}=Val(256))
    update!(enc, x, y, [z], ::Val{ThreadsPerGroup}=Val(256))

Resize and rebuild a Metal-backed `MortonEncoding` from new unsorted
coordinates. Existing device vectors are resized, then `build!` copies the
coordinates and recomputes Morton codes. The result remains unsorted.

# Parameters
- `enc`: Reusable Metal Morton encoding.
- `coords`: Metal coordinate vectors in structure-of-arrays form.
- `x`, `y`, `z`: Coordinate-wise convenience arguments.
- `ThreadsPerGroup`: Number of threads per Metal threadgroup.

# Returns
- `MortonEncoding`: The same `enc` wrapper with resized and updated storage.
"""
function Partia.update!(enc :: MortonEncoding{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, coords :: NTuple{D, MtlVector{Float32}}, :: Val{ThreadsPerGroup} = Val(256)) where {D, ThreadsPerGroup, TI <: Unsigned}
    D in (2, 3) || throw(ArgumentError("Morton encoding only supports two or three dimensions"))
    n = length(coords[1])
    n > 0 || throw(ArgumentError("coordinates must not be empty"))
    all(length(coord) == n for coord in coords) || throw(DimensionMismatch("coordinates must have identical lengths"))

    resize!(enc.order, n)
    resize!(enc.codes, n)
    @inbounds for d in 1:D
        resize!(enc.coord[d], n)
    end

    Partia.build!(enc, coords, Val(ThreadsPerGroup))
    return enc
end

function Partia.update!(enc :: MortonEncoding{2, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, x :: MtlVector{Float32}, y :: MtlVector{Float32}, args...) where {TI <: Unsigned}
    return Partia.update!(enc, (x, y), args...)
end

function Partia.update!(enc :: MortonEncoding{3, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, x :: MtlVector{Float32}, y :: MtlVector{Float32}, z :: MtlVector{Float32}, args...) where {TI <: Unsigned}
    return Partia.update!(enc, (x, y, z), args...)
end

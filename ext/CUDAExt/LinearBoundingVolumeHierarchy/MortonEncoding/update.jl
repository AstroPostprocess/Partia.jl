"""
    update!(enc, coords, [launch configuration])
    update!(enc, x, y, [z], [launch configuration])

Resize and rebuild a CUDA-backed `MortonEncoding` from new unsorted
coordinates. Existing device vectors are resized, then `build!` copies the
coordinates and recomputes Morton codes. The result remains unsorted.

# Parameters
- `enc`: Reusable CUDA Morton encoding.
- `coords`: CUDA coordinate vectors in structure-of-arrays form.
- `x`, `y`, `z`: Coordinate-wise convenience arguments.
- `NBlocks`: Number of CUDA blocks.
- `ThreadsPerBlock`: Number of threads per CUDA block.

# Returns
- `MortonEncoding`: The same `enc` wrapper with resized and updated storage.
"""
function Partia.update!(enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, coords :: NTuple{D, CuVector{TF}}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned}
    D in (2, 3) || throw(ArgumentError("Morton encoding only supports two or three dimensions"))
    n = length(coords[1])
    n > 0 || throw(ArgumentError("coordinates must not be empty"))
    all(length(coord) == n for coord in coords) || throw(DimensionMismatch("coordinates must have identical lengths"))

    resize!(enc.order, n)
    resize!(enc.codes, n)
    @inbounds for d in 1:D
        resize!(enc.coord[d], n)
    end

    Partia.build!(enc, coords, Val(NBlocks), Val(ThreadsPerBlock))
    return enc
end

function Partia.update!(enc :: MortonEncoding{2, TF, TI, CuVector{TF}, CuVector{TI}}, x :: CuVector{TF}, y :: CuVector{TF}, args...) where {TF <: AbstractFloat, TI <: Unsigned}
    return Partia.update!(enc, (x, y), args...)
end

function Partia.update!(enc :: MortonEncoding{3, TF, TI, CuVector{TF}, CuVector{TI}}, x :: CuVector{TF}, y :: CuVector{TF}, z :: CuVector{TF}, args...) where {TF <: AbstractFloat, TI <: Unsigned}
    return Partia.update!(enc, (x, y, z), args...)
end

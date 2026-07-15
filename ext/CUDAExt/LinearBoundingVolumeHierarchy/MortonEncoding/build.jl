######################################################################################

# Reusable and no-copy Morton encoding builds for CUDA.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
"""
    build!(enc, coords,
           ::Val{NBlocks}=Val(256),
           ::Val{ThreadsPerBlock}=Val(256))
    build!(enc, x, y, [z],
           ::Val{NBlocks}=Val(256),
           ::Val{ThreadsPerBlock}=Val(256))

Recompute a preallocated CUDA Morton encoding from unsorted coordinates while
reusing its code and coordinate storage. Input coordinates are copied into
`enc.coord` before the encoding kernel is launched. This function does not sort
the result; call `sort_by_morton!` before constructing a `LinearBVH`.

# Parameters
- `enc`: Preallocated CUDA `MortonEncoding` to rebuild.
- `coords`: Two- or three-dimensional coordinates in structure-of-arrays form.
- `x`, `y`, `z`: Coordinate-wise convenience arguments for `coords`.
- `NBlocks`: Number of CUDA blocks used by the encoding kernel. Defaults to 256.
- `ThreadsPerBlock`: Number of threads per CUDA block. Defaults to 256.

# Returns
- `nothing`: `enc.codes` and `enc.coord` are updated in place in input order.
"""
function Partia.build!(enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, coords :: NTuple{D, CuVector{TF}}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned}
    D in (2, 3) || throw(ArgumentError("Morton encoding only supports two or three dimensions"))
    n = length(enc.codes)
    n > 0 || throw(ArgumentError("coordinates must not be empty"))
    length(enc.order) == n || throw(DimensionMismatch("enc.order and enc.codes must have identical lengths"))
    all(length(coord) == n for coord in coords) || throw(DimensionMismatch("coords and enc.codes must have identical lengths"))
    all(axes(coord) == axes(coords[1]) for coord in coords) || throw(DimensionMismatch("coordinates must have identical axes"))

    # Restore unsorted inputs into the reusable coordinate storage.
    for d in 1:D
        copyto!(enc.coord[d], coords[d])
    end

    bounds = Partia.LinearBoundingVolumeHierarchy._coordinate_bounds(coords)
    inv_extent = ntuple(D) do d
        extent = bounds[d][2] - bounds[d][1]
        iszero(extent) ? zero(TF) : inv(extent)
    end
    offset = ntuple(D) do d
        iszero(inv_extent[d]) ? TF(0.5) : -inv_extent[d] * bounds[d][1]
    end

    @cuda threads=ThreadsPerBlock blocks=NBlocks Partia.LinearBoundingVolumeHierarchy._morton_encoding_kernel!(enc.codes, enc.coord, inv_extent, offset)
    return nothing
end

function Partia.build!(enc :: MortonEncoding{2, TF, TI, CuVector{TF}, CuVector{TI}}, x :: CuVector{TF}, y :: CuVector{TF}, args...) where {TF <: AbstractFloat, TI <: Unsigned}
    Partia.build!(enc, (x, y), args...)
    return nothing
end

function Partia.build!(enc :: MortonEncoding{3, TF, TI, CuVector{TF}, CuVector{TI}}, x :: CuVector{TF}, y :: CuVector{TF}, z :: CuVector{TF}, args...) where {TF <: AbstractFloat, TI <: Unsigned}
    Partia.build!(enc, (x, y, z), args...)
    return nothing
end

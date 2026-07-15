######################################################################################

# Reusable and no-copy Morton encoding builds for Metal.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
"""
    build!(enc, points, ::Val{ThreadsPerGroup}=Val(256))
    build!(enc, x, y, [z], ::Val{ThreadsPerGroup}=Val(256))

Recompute a preallocated Metal Morton encoding from unsorted coordinates while
reusing its code and coordinate storage. Input coordinates are copied into
`enc.coord` before the encoding kernel is launched. This function does not sort
the result; call `sort_by_morton!` before constructing a `LinearBVH`.

# Parameters
- `enc`: Preallocated Metal `MortonEncoding` to rebuild.
- `points`: Two- or three-dimensional coordinates in structure-of-arrays form.
- `x`, `y`, `z`: Coordinate-wise convenience arguments for `points`.
- `ThreadsPerGroup`: Number of threads per Metal threadgroup. Defaults to 256.

# Returns
- `nothing`: `enc.codes` and `enc.coord` are updated in place in input order.
"""
function Partia.build!(enc :: MortonEncoding{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, points :: NTuple{D, MtlVector{Float32}}, :: Val{ThreadsPerGroup} = Val(256)) where {D, ThreadsPerGroup, TI <: Unsigned}
    D in (2, 3) || throw(ArgumentError("Morton encoding only supports two or three dimensions"))
    n = length(enc.codes)
    n > 0 || throw(ArgumentError("coordinates must not be empty"))
    length(enc.order) == n || throw(DimensionMismatch("enc.order and enc.codes must have identical lengths"))
    all(length(p) == n for p in points) || throw(DimensionMismatch("points and enc.codes must have identical lengths"))
    all(axes(p) == axes(points[1]) for p in points) || throw(DimensionMismatch("coordinates must have identical axes"))

    # Restore unsorted inputs into the reusable coordinate storage.
    for d in 1:D
        copyto!(enc.coord[d], points[d])
    end

    bounds = Partia.LinearBoundingVolumeHierarchy._coordinate_bounds(points)
    inv_extent = ntuple(D) do d
        extent = bounds[d][2] - bounds[d][1]
        iszero(extent) ? zero(Float32) : inv(extent)
    end
    offset = ntuple(D) do d
        iszero(inv_extent[d]) ? Float32(0.5) : -inv_extent[d] * bounds[d][1]
    end

    @metal threads=(ThreadsPerGroup,) groups=(cld(n, ThreadsPerGroup),) Partia.LinearBoundingVolumeHierarchy._morton_encoding_kernel!(enc.codes, enc.coord, inv_extent, offset)
    return nothing
end

function Partia.build!(enc :: MortonEncoding{2, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, x :: MtlVector{Float32}, y :: MtlVector{Float32}, args...) where {TI <: Unsigned}
    Partia.build!(enc, (x, y), args...)
    return nothing
end

function Partia.build!(enc :: MortonEncoding{3, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, x :: MtlVector{Float32}, y :: MtlVector{Float32}, z :: MtlVector{Float32}, args...) where {TI <: Unsigned}
    Partia.build!(enc, (x, y, z), args...)
    return nothing
end

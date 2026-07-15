######################################################################################

# Reusable and no-copy Morton encoding builders.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
"""
    build!(enc, points, workspace, ::Val{TileSize}=Val(8192))

Recompute a preallocated Morton encoding from unsorted structure-of-arrays
coordinates. Codes, order, coordinate storage, and OneSweep workspace are all
reused. Input coordinates are copied into `enc.coord` before in-place sorting.

# Returns
- `enc`: The recomputed encoding in Morton order.
"""
function build!(enc :: MortonEncoding{D, TF, TI, Vector{TF}, Vector{TI}}, points :: NTuple{D, Vector{TF}}, workspace :: OnesweepWorkspace{TI, Vector{TI}, Vector{UInt32}}, :: Val{TileSize} = Val(8192)) where {D, TileSize, TF <: AbstractFloat, TI <: Unsigned}
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
    sort_by_morton!(enc, workspace, Val(TileSize))
    return nothing
end

function build!(enc :: MortonEncoding{2, TF, TI, Vector{TF}, Vector{TI}}, x :: Vector{TF}, y :: Vector{TF}, workspace :: OnesweepWorkspace{TI, Vector{TI}, Vector{UInt32}}, args...) where {TF <: AbstractFloat, TI <: Unsigned}
    build!(enc, (x, y), workspace, args...)
    return nothing
end

function build!(enc :: MortonEncoding{3, TF, TI, Vector{TF}, Vector{TI}}, x :: Vector{TF}, y :: Vector{TF}, z :: Vector{TF}, workspace :: OnesweepWorkspace{TI, Vector{TI}, Vector{UInt32}}, args...) where {TF <: AbstractFloat, TI <: Unsigned}
    build!(enc, (x, y, z), workspace, args...)
    return nothing
end
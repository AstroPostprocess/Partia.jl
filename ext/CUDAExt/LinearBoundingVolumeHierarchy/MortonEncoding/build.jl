######################################################################################

# Reusable and no-copy Morton encoding builds for CUDA.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
"""
    build!(enc, points, workspace,
           ::Val{TileSize}=Val(4096),
           ::Val{NBlocks}=Val(256),
           ::Val{ThreadsPerBlock}=Val(256))

Recompute a preallocated CUDA Morton encoding from unsorted coordinates while
reusing codes, order, coordinate storage, and the OneSweep workspace.
"""
function Partia.build!(enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, points :: NTuple{D, CuVector{TF}}, workspace :: OnesweepWorkspace{TI, CodeV, OffsetV}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, TileSize, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned, CodeV <: CuVector{TI}, OffsetV <: CuVector{UInt32}}
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
        iszero(extent) ? zero(TF) : inv(extent)
    end
    offset = ntuple(D) do d
        iszero(inv_extent[d]) ? TF(0.5) : -inv_extent[d] * bounds[d][1]
    end

    @cuda threads=ThreadsPerBlock blocks=NBlocks Partia.LinearBoundingVolumeHierarchy._morton_encoding_kernel!(enc.codes, enc.coord, inv_extent, offset)
    Partia.sort_by_morton!(enc, workspace, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock))
    return nothing
end

function Partia.build!(enc :: MortonEncoding{2, TF, TI, CuVector{TF}, CuVector{TI}}, x :: CuVector{TF}, y :: CuVector{TF}, workspace, args...) where {TF <: AbstractFloat, TI <: Unsigned}
    Partia.build!(enc, (x, y), workspace, args...)
    return nothing
end

function Partia.build!(enc :: MortonEncoding{3, TF, TI, CuVector{TF}, CuVector{TI}}, x :: CuVector{TF}, y :: CuVector{TF}, z :: CuVector{TF}, workspace, args...) where {TF <: AbstractFloat, TI <: Unsigned}
    Partia.build!(enc, (x, y, z), workspace, args...)
    return nothing
end

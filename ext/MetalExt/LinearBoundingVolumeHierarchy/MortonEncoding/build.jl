######################################################################################

# Reusable and no-copy Morton encoding builds for Metal.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
"""
    build!(enc, points, workspace,
           ::Val{TileSize}=Val(2048),
           ::Val{NThreadgroups}=Val(128),
           ::Val{ThreadsPerGroup}=Val(256))

Recompute a preallocated Metal Morton encoding from unsorted coordinates while
reusing codes, order, coordinate storage, and the OneSweep workspace.
"""
function Partia.build!(enc :: MortonEncoding{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, points :: NTuple{D, MtlVector{Float32}}, workspace :: OnesweepWorkspace{TI, CodeV, OffsetV}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {D, TileSize, NThreadgroups, ThreadsPerGroup, TI <: Unsigned, CodeV <: MtlVector{TI}, OffsetV <: MtlVector{UInt32}}
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
    Partia.sort_by_morton!(enc, workspace, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))
    return enc
end

function Partia.build!(enc :: MortonEncoding{2, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, x :: MtlVector{Float32}, y :: MtlVector{Float32}, workspace, args...) where {TI <: Unsigned}
    return Partia.build!(enc, (x, y), workspace, args...)
end

function Partia.build!(enc :: MortonEncoding{3, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, x :: MtlVector{Float32}, y :: MtlVector{Float32}, z :: MtlVector{Float32}, workspace, args...) where {TI <: Unsigned}
    return Partia.build!(enc, (x, y, z), workspace, args...)
end

"""Metal no-copy Morton encoding; input coordinate vectors are sorted in place."""
function Partia.MortonEncoding!(x :: MtlVector{Float32}, y :: MtlVector{Float32}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256); CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(MtlVector{CodeType})) where {TileSize, NThreadgroups, ThreadsPerGroup, TI <: Unsigned}
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    axes(x) == axes(y) || throw(DimensionMismatch("x and y must have identical axes"))
    codes = MtlVector{TI}(undef, length(x))
    enc = Partia.MortonEncoding{2, Float32, TI, MtlVector{Float32}, MtlVector{TI}}(similar(codes), codes, (x, y))
    return Partia.build!(enc, x, y, SortWorkSpace, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))
end

function Partia.MortonEncoding!(x :: MtlVector{Float32}, y :: MtlVector{Float32}, z :: MtlVector{Float32}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256); CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(MtlVector{CodeType})) where {TileSize, NThreadgroups, ThreadsPerGroup, TI <: Unsigned}
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    axes(x) == axes(y) == axes(z) || throw(DimensionMismatch("x, y, and z must have identical axes"))
    codes = MtlVector{TI}(undef, length(x))
    enc = Partia.MortonEncoding{3, Float32, TI, MtlVector{Float32}, MtlVector{TI}}(similar(codes), codes, (x, y, z))
    return Partia.build!(enc, x, y, z, SortWorkSpace, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))
end

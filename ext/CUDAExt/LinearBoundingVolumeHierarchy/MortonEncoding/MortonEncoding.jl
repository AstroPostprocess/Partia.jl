######################################################################################

# Morton encoding constructors for particle spatial indexing with CUDA.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
"""
    MortonEncoding(points::NTuple{D,CuVector{TF}},
                   ::Val{TileSize}=Val(4096),
                   ::Val{NBlocks}=Val(256),
                   ::Val{ThreadsPerBlock}=Val(256);
                   CodeType=UInt64,
                   SortWorkSpace=OnesweepWorkspace(CuVector{CodeType}))

Encode two- or three-dimensional CUDA coordinates into Morton codes and sort
the coordinates into Morton order. `TF` may be any CUDA-supported subtype of
`AbstractFloat`, and `D` must be either 2 or 3.

# Parameters
- `points`: Structure-of-arrays coordinates `(x, y)` or `(x, y, z)`.
- `TileSize`: Compile-time OneSweep radix-sort tile size. Defaults to 4096.
- `NBlocks`: Number of CUDA blocks used by encoding and sorting. Defaults to 256.
- `ThreadsPerBlock`: Threads in each CUDA block. Defaults to 256.

# Keyword Arguments
- `CodeType`: Unsigned Morton-code type, normally `UInt32` or `UInt64`.
- `SortWorkSpace`: Reusable CUDA OneSweep workspace.

# Returns
A `MortonEncoding` whose codes and copied coordinates are in Morton order and
whose `order` field maps the original particle order to that ordering.
"""
function Partia.MortonEncoding(points :: NTuple{D, CuVector{TF}}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256);
    CodeType :: Type{TI} = UInt64,
    SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(CuVector{CodeType})) where {D, TF <: AbstractFloat, TileSize, NBlocks, ThreadsPerBlock, TI <: Unsigned}
    # Validate the structure-of-arrays input before launching a GPU kernel.
    D in (2, 3) || throw(ArgumentError("Morton encoding only supports two or three dimensions"))
    all(!isempty, points) || throw(ArgumentError("coordinates must not be empty"))
    all(axes(p) == axes(points[1]) for p in points) || throw(DimensionMismatch("coordinates must have identical axes"))

    # Preserve the caller's coordinate arrays: sorting below is in-place.
    coord = map(copy, points)

    # Build an affine map from each physical coordinate range to [0, 1].
    # A degenerate axis is placed at the midpoint so all equal coordinates
    # receive the same stable quantized value.
    bounds = map(extrema, coord)
    inv_extent = ntuple(D) do d
        extent = bounds[d][2] - bounds[d][1]
        iszero(extent) ? zero(TF) : inv(extent)
    end
    offset = ntuple(D) do d
        iszero(inv_extent[d]) ? TF(0.5) : -inv_extent[d] * bounds[d][1]
    end

    # Encode directly from physical coordinates, avoiding temporary normalized
    # and quantized coordinate vectors.
    codes = CuVector{TI}(undef, length(coord[1]))
    order = similar(codes)
    @cuda threads=ThreadsPerBlock blocks=NBlocks Partia.LinearBoundingVolumeHierarchy._morton_encoding_kernel!(codes, coord, inv_extent, offset)
    CUDA.synchronize()

    # Sort codes and coordinate copies together and retain the permutation.
    enc = Partia.MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}(order, codes, coord)
    Partia.sort_by_morton!(enc, SortWorkSpace, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock))
    return enc
end

"""Coordinate-wise 2D convenience overload for CUDA Morton encoding."""
function Partia.MortonEncoding(x :: CuVector{TF}, y :: CuVector{TF}, args...; kwargs...) where {TF <: AbstractFloat}
    return Partia.MortonEncoding((x, y), args...; kwargs...)
end

"""Coordinate-wise 3D convenience overload for CUDA Morton encoding."""
function Partia.MortonEncoding(x :: CuVector{TF}, y :: CuVector{TF}, z :: CuVector{TF}, args...; kwargs...) where {TF <: AbstractFloat}
    return Partia.MortonEncoding((x, y, z), args...; kwargs...)
end

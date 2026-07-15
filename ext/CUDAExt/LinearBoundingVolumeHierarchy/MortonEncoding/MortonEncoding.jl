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
    SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(CuVector{CodeType})) where {D, TileSize, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned}
    # Validate the structure-of-arrays input before launching a GPU kernel.
    D in (2, 3) || throw(ArgumentError("Morton encoding only supports two or three dimensions"))
    all(!isempty, points) || throw(ArgumentError("coordinates must not be empty"))
    all(axes(p) == axes(points[1]) for p in points) || throw(DimensionMismatch("coordinates must have identical axes"))

    # Allocate reusable encoding storage, then populate it through the in-place path.
    coord = map(similar, points)
    codes = CuVector{TI}(undef, length(points[1]))
    order = similar(codes)
    enc = Partia.MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}(order, codes, coord)
    Partia.build!(enc, points, SortWorkSpace, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock))
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

"""CUDA no-copy Morton encoding; input coordinate vectors are sorted in place."""
function Partia.MortonEncoding!(x :: CuVector{TF}, y :: CuVector{TF}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256); CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(CuVector{CodeType})) where {TileSize, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned}
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    axes(x) == axes(y) || throw(DimensionMismatch("x and y must have identical axes"))
    codes = CuVector{TI}(undef, length(x))
    enc = Partia.MortonEncoding{2, TF, TI, CuVector{TF}, CuVector{TI}}(similar(codes), codes, (x, y))
    Partia.build!(enc, x, y, SortWorkSpace, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock))
    return enc
end

function Partia.MortonEncoding!(x :: CuVector{TF}, y :: CuVector{TF}, z :: CuVector{TF}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256); CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(CuVector{CodeType})) where {TileSize, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned}
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    axes(x) == axes(y) == axes(z) || throw(DimensionMismatch("x, y, and z must have identical axes"))
    codes = CuVector{TI}(undef, length(x))
    enc = Partia.MortonEncoding{3, TF, TI, CuVector{TF}, CuVector{TI}}(similar(codes), codes, (x, y, z))
    Partia.build!(enc, x, y, z, SortWorkSpace, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock))
    return enc
end

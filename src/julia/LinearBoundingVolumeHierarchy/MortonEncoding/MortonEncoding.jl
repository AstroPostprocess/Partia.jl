######################################################################################

# Morton encoding data structure and constructors for particle spatial indexing.
#     by Wei-Shan Su,
#     July 12, 2026

######################################################################################
################# Define structures #################
struct MortonEncoding{D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}}
    order :: VI             # Order of corresponding particles
    codes :: VI             # Morton code
    coord :: NTuple{D, VF}  # Original data points
end

"""Return per-axis coordinate bounds using scalar reductions."""
@inline function _coordinate_bounds(points :: NTuple{D, V}) where {D, V <: AbstractVector}
    return map(point -> (minimum(point), maximum(point)), points)
end

function Adapt.adapt_structure(to, x :: ME) where {D, ME <: MortonEncoding{D}}
    MortonEncoding(
        Adapt.adapt(to, x.order),
        Adapt.adapt(to, x.codes),
        ntuple(i -> Adapt.adapt(to, x.coord[i]), D)
    )
end

################# Encoding Morton code #################
"""
    MortonEncoding(x::Vector{T}, y::Vector{T}, z::Vector{T}, ::Val{TileSize}=Val(4096);
                   CodeType=UInt64,
                   SortWorkSpace=OnesweepWorkspace(Vector{CodeType}))

Encode a set of 3D particle coordinates into Morton codes.

# Parameters
- `x, y, z :: Vector{T}`: Particle positions along each axis (floating-point).
- `::Val{TileSize}`: Compile-time tile size used by the OneSweep radix sorter.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |
| `SortWorkSpace` | `OnesweepWorkspace{TI}` | `OnesweepWorkspace(Vector{CodeType})` | Reusable workspace for Morton-code sorting. |

# Returns
- `MortonEncoding{3, T, TI, Vector{T}, Vector{TI}}`: Encoding containing Morton codes,
  original particle indices, and copied coordinates, all ordered by Morton code.
"""
function MortonEncoding(x :: Vector{T}, y :: Vector{T}, z :: Vector{T}, :: Val{TileSize} = Val(4096); CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(Vector{CodeType})) where {TileSize, TI <: Unsigned, T <: AbstractFloat}
    # Verify length of input arrays
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    isempty(y) && throw(ArgumentError("coordinates must not be empty"))
    isempty(z) && throw(ArgumentError("coordinates must not be empty"))

    axes(x) == axes(y) == axes(z) || throw(DimensionMismatch("x, y, and z must have identical axes"))

    # Copy the coordinates to prevent modification
    xcopy = copy(x); ycopy = copy(y); zcopy = copy(z)
    npart = length(xcopy)

    # Compute each bound with a scalar reduction. In particular, this avoids
    # tuple-valued `extrema` reductions on accelerator backends.
    bounds = _coordinate_bounds((xcopy, ycopy, zcopy))
    xmin, xmax = bounds[1]
    ymin, ymax = bounds[2]
    zmin, zmax = bounds[3]

    # Total length of the box
    Δx = xmax - xmin
    Δy = ymax - ymin
    Δz = zmax - zmin

    # Compute the inverse extent and normalization offset.
    # Map every coordinate on a degenerate axis to the midpoint, 0.5.
    degeneratex = iszero(Δx)
    degeneratey = iszero(Δy)
    degeneratez = iszero(Δz)

    invΔx = degeneratex ? zero(T) : inv(Δx)
    invΔy = degeneratey ? zero(T) : inv(Δy)
    invΔz = degeneratez ? zero(T) : inv(Δz)

    # Prepare the normalization offsets (fx = invΔx * xi + cx).
    cx = degeneratex ? T(0.5) : -invΔx * xmin
    cy = degeneratey ? T(0.5) : -invΔy * ymin
    cz = degeneratez ? T(0.5) : -invΔz * zmin

    # Allocate vectors for the final result
    codes = Vector{TI}(undef, npart)
    order = similar(codes)

    # Encode all points without allocating normalized or quantized coordinates
    @inbounds @threads for i in eachindex(codes, xcopy, ycopy, zcopy)
        _morton_encoding_kernel!(codes, i, (xcopy, ycopy, zcopy), (invΔx, invΔy, invΔz), (cx, cy, cz))
    end

    # Construct structure
    enc = MortonEncoding{3, T, TI, Vector{T}, Vector{TI}}(order, codes, (xcopy, ycopy, zcopy))

    # Sort by morton
    sort_by_morton!(enc, SortWorkSpace, Val(TileSize))
    return enc
end

"""
    MortonEncoding(points::NTuple{3,Vector{T}}, ::Val{TileSize}=Val(4096);
                   CodeType=UInt64,
                   SortWorkSpace=OnesweepWorkspace(Vector{CodeType}))

Encode a set of 3D particle coordinates into Morton codes.

This overload accepts particle positions in a structure-of-arrays (SoA) layout,
where `points = (x, y, z)`. It forwards to
`MortonEncoding(x, y, z; CodeType=CodeType)`.

# Parameters
- `points :: NTuple{3,Vector{T}}`: Particle coordinates stored as `(x, y, z)`.
- `::Val{TileSize}`: Compile-time tile size used by the OneSweep radix sorter.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |
| `SortWorkSpace` | `OnesweepWorkspace{TI}` | `OnesweepWorkspace(Vector{CodeType})` | Reusable workspace for Morton-code sorting. |

# Returns
- A 3D `MortonEncoding` with codes, indices, and coordinates ordered by Morton code.
"""
function MortonEncoding(points :: NTuple{3, Vector{T}}, :: Val{TileSize} = Val(4096); CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(Vector{CodeType})) where {TileSize, TI <: Unsigned, T <: AbstractFloat}
    x = points[1]; y = points[2]; z = points[3]
    return MortonEncoding(x, y, z, Val(TileSize); CodeType, SortWorkSpace)
end

"""
    MortonEncoding(x::Vector{T}, y::Vector{T}, ::Val{TileSize}=Val(4096);
                   CodeType=UInt64,
                   SortWorkSpace=OnesweepWorkspace(Vector{CodeType}))

Encode a set of 2D particle coordinates into Morton codes.

# Parameters
- `x, y :: Vector{T}`: Particle positions along each axis (floating-point).
- `::Val{TileSize}`: Compile-time tile size used by the OneSweep radix sorter.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |
| `SortWorkSpace` | `OnesweepWorkspace{TI}` | `OnesweepWorkspace(Vector{CodeType})` | Reusable workspace for Morton-code sorting. |

# Returns
- `MortonEncoding{2, T, TI, Vector{T}, Vector{TI}}`: Encoding containing Morton codes,
  original particle indices, and copied coordinates, all ordered by Morton code.
"""
function MortonEncoding(x :: Vector{T}, y :: Vector{T}, :: Val{TileSize} = Val(4096); CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(Vector{CodeType})) where {TileSize, TI <: Unsigned, T <: AbstractFloat}
    # Verify length of input arrays
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    isempty(y) && throw(ArgumentError("coordinates must not be empty"))

    axes(x) == axes(y) || throw(DimensionMismatch("x and y must have identical axes"))

    # Copy the coordinates to prevent modification
    xcopy = copy(x); ycopy = copy(y)
    npart = length(xcopy)

    # Compute each bound with a scalar reduction. In particular, this avoids
    # tuple-valued `extrema` reductions on accelerator backends.
    bounds = _coordinate_bounds((xcopy, ycopy))
    xmin, xmax = bounds[1]
    ymin, ymax = bounds[2]

    # Total length of the box
    Δx = xmax - xmin
    Δy = ymax - ymin

    # Compute the inverse extent and normalization offset.
    # Map every coordinate on a degenerate axis to the midpoint, 0.5.
    degeneratex = iszero(Δx)
    degeneratey = iszero(Δy)

    invΔx = degeneratex ? zero(T) : inv(Δx)
    invΔy = degeneratey ? zero(T) : inv(Δy)

    # Prepare the normalization offsets (fx = invΔx * xi + cx).
    cx = degeneratex ? T(0.5) : -invΔx * xmin
    cy = degeneratey ? T(0.5) : -invΔy * ymin

    # Allocate vectors for the final result
    codes = Vector{TI}(undef, npart)
    order = similar(codes)

    # Encode all points without allocating normalized or quantized coordinates
    @inbounds @threads for i in eachindex(codes, xcopy, ycopy)
        _morton_encoding_kernel!(codes, i, (xcopy, ycopy), (invΔx, invΔy), (cx, cy))
    end

    # Construct structure
    enc = MortonEncoding{2, T, TI, Vector{T}, Vector{TI}}(order, codes, (xcopy, ycopy))

    # Sort by morton
    sort_by_morton!(enc, SortWorkSpace, Val(TileSize))
    return enc
end

"""
    MortonEncoding(points::NTuple{2,Vector{T}}, ::Val{TileSize}=Val(4096);
                   CodeType=UInt64,
                   SortWorkSpace=OnesweepWorkspace(Vector{CodeType}))

Encode a set of 2D particle coordinates into Morton codes.

This overload accepts particle positions in a structure-of-arrays (SoA) layout,
where `points = (x, y)`. It forwards to
`MortonEncoding(x, y; CodeType=CodeType)`.

# Parameters
- `points :: NTuple{2,Vector{T}}`: Particle coordinates stored as `(x, y)`.
- `::Val{TileSize}`: Compile-time tile size used by the OneSweep radix sorter.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |
| `SortWorkSpace` | `OnesweepWorkspace{TI}` | `OnesweepWorkspace(Vector{CodeType})` | Reusable workspace for Morton-code sorting. |

# Returns
- A 2D `MortonEncoding` with codes, indices, and coordinates ordered by Morton code.
"""
function MortonEncoding(points :: NTuple{2, Vector{T}}, :: Val{TileSize} = Val(4096); CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(Vector{CodeType})) where {TileSize, TI <: Unsigned, T <: AbstractFloat}
    x = points[1]; y = points[2]
    return MortonEncoding(x, y, Val(TileSize); CodeType, SortWorkSpace)
end

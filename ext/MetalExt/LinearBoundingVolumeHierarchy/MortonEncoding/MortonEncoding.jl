######################################################################################

# Morton encoding constructors for particle spatial indexing with Metal.
#     by Wei-Shan Su,
#     July 13, 2026

######################################################################################
################# Encoding Morton code #################
"""
    MortonEncoding(x::MtlVector{Float32}, y::MtlVector{Float32}, z::MtlVector{Float32}, ::Val{TileSize}=Val(2048), ::Val{NThreadgroups}=Val(128), ::Val{ThreadsPerGroup}=Val(256);
                   CodeType=UInt64,
                   SortWorkSpace=OnesweepWorkspace(MtlVector{CodeType}))

Encode a set of 3D particle coordinates into Morton codes.

# Parameters
- `x, y, z :: MtlVector{Float32}`: Particle positions along each axis (floating-point).
- `::Val{TileSize}`: Compile-time tile size used by the OneSweep radix sorter.
- `::Val{NThreadgroups}`: Number of threadgroups used by the radix sorter.
- `::Val{ThreadsPerGroup}`: Number of threads in each Metal threadgroup.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |
| `SortWorkSpace` | `OnesweepWorkspace{TI}` | `OnesweepWorkspace(MtlVector{CodeType})` | Reusable workspace for Morton-code sorting. |

# Returns
- `MortonEncoding{3, Float32, TI, MtlVector{Float32}, MtlVector{TI}}`: Encoding containing Morton codes,
  original particle indices, and copied coordinates, all ordered by Morton code.
"""
function Partia.MortonEncoding(x :: MtlVector{Float32}, y :: MtlVector{Float32}, z :: MtlVector{Float32}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256);
    CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(MtlVector{CodeType})) where {TileSize, NThreadgroups, ThreadsPerGroup, TI <: Unsigned}
    # Verify length of input arrays
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    isempty(y) && throw(ArgumentError("coordinates must not be empty"))
    isempty(z) && throw(ArgumentError("coordinates must not be empty"))

    axes(x) == axes(y) == axes(z) || throw(DimensionMismatch("x, y, and z must have identical axes"))

    # Copy the coordinates to prevent modification
    xcopy = copy(x); ycopy = copy(y); zcopy = copy(z)
    npart = length(xcopy)

    # Get the extrema for each axis
    xmin, xmax = extrema(xcopy)
    ymin, ymax = extrema(ycopy)
    zmin, zmax = extrema(zcopy)

    # Total length of the box
    Δx = xmax - xmin
    Δy = ymax - ymin
    Δz = zmax - zmin

    # Compute the inverse extent and normalization offset.
    # Map every coordinate on a degenerate axis to the midpoint, 0.5.
    degeneratex = iszero(Δx)
    degeneratey = iszero(Δy)
    degeneratez = iszero(Δz)

    invΔx = degeneratex ? zero(Float32) : inv(Δx)
    invΔy = degeneratey ? zero(Float32) : inv(Δy)
    invΔz = degeneratez ? zero(Float32) : inv(Δz)

    # Prepare the normalization offsets (fx = invΔx * xi + cx).
    cx = degeneratex ? Float32(0.5) : -invΔx * xmin
    cy = degeneratey ? Float32(0.5) : -invΔy * ymin
    cz = degeneratez ? Float32(0.5) : -invΔz * zmin

    # Allocate vectors for the final result
    codes = MtlVector{TI}(undef, npart)
    order = similar(codes)

    # Encode all points without allocating normalized or quantized coordinates
    @metal threads=(ThreadsPerGroup,) groups=(cld(npart, ThreadsPerGroup),) Partia.LinearBoundingVolumeHierarchy._morton_encoding_kernel!(codes, (xcopy, ycopy, zcopy), (invΔx, invΔy, invΔz), (cx, cy, cz))
    Metal.synchronize()

    # Construct structure
    enc = Partia.MortonEncoding{3, Float32, TI, MtlVector{Float32}, MtlVector{TI}}(order, codes, (xcopy, ycopy, zcopy))

    # Sort by morton
    Partia.sort_by_morton!(enc, SortWorkSpace, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))
    return enc
end

"""
    MortonEncoding(points::NTuple{3,MtlVector{Float32}}, ::Val{TileSize}=Val(2048), ::Val{NThreadgroups}=Val(128), ::Val{ThreadsPerGroup}=Val(256);
                   CodeType=UInt64,
                   SortWorkSpace=OnesweepWorkspace(MtlVector{CodeType}))

Encode a set of 3D particle coordinates into Morton codes.

This overload accepts particle positions in a structure-of-arrays (SoA) layout,
where `points = (x, y, z)`. It forwards to
`MortonEncoding(x, y, z; CodeType=CodeType)`.

# Parameters
- `points :: NTuple{3,MtlVector{Float32}}`: Particle coordinates stored as `(x, y, z)`.
- `::Val{TileSize}`: Compile-time tile size used by the OneSweep radix sorter.
- `::Val{NThreadgroups}`: Number of threadgroups used by the radix sorter.
- `::Val{ThreadsPerGroup}`: Number of threads in each Metal threadgroup.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |
| `SortWorkSpace` | `OnesweepWorkspace{TI}` | `OnesweepWorkspace(MtlVector{CodeType})` | Reusable workspace for Morton-code sorting. |

# Returns
- A 3D `MortonEncoding` with codes, indices, and coordinates ordered by Morton code.
"""
function Partia.MortonEncoding(points :: NTuple{3, MtlVector{Float32}}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256);
    CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(MtlVector{CodeType})) where {TileSize, NThreadgroups, ThreadsPerGroup, TI <: Unsigned}
    x = points[1]; y = points[2]; z = points[3]
    return Partia.MortonEncoding(x, y, z, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup); CodeType, SortWorkSpace)
end

"""
    MortonEncoding(x::MtlVector{Float32}, y::MtlVector{Float32}, ::Val{TileSize}=Val(2048), ::Val{NThreadgroups}=Val(128), ::Val{ThreadsPerGroup}=Val(256);
                   CodeType=UInt64,
                   SortWorkSpace=OnesweepWorkspace(MtlVector{CodeType}))

Encode a set of 2D particle coordinates into Morton codes.

# Parameters
- `x, y :: MtlVector{Float32}`: Particle positions along each axis (floating-point).
- `::Val{TileSize}`: Compile-time tile size used by the OneSweep radix sorter.
- `::Val{NThreadgroups}`: Number of threadgroups used by the radix sorter.
- `::Val{ThreadsPerGroup}`: Number of threads in each Metal threadgroup.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |
| `SortWorkSpace` | `OnesweepWorkspace{TI}` | `OnesweepWorkspace(MtlVector{CodeType})` | Reusable workspace for Morton-code sorting. |

# Returns
- `MortonEncoding{2, Float32, TI, MtlVector{Float32}, MtlVector{TI}}`: Encoding containing Morton codes,
  original particle indices, and copied coordinates, all ordered by Morton code.
"""
function Partia.MortonEncoding(x :: MtlVector{Float32}, y :: MtlVector{Float32}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256);
    CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(MtlVector{CodeType})) where {TileSize, NThreadgroups, ThreadsPerGroup, TI <: Unsigned}
    # Verify length of input arrays
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    isempty(y) && throw(ArgumentError("coordinates must not be empty"))

    axes(x) == axes(y) || throw(DimensionMismatch("x and y must have identical axes"))

    # Copy the coordinates to prevent modification
    xcopy = copy(x); ycopy = copy(y)
    npart = length(xcopy)

    # Get the extrema for each axis
    xmin, xmax = extrema(xcopy)
    ymin, ymax = extrema(ycopy)

    # Total length of the box
    Δx = xmax - xmin
    Δy = ymax - ymin

    # Compute the inverse extent and normalization offset.
    # Map every coordinate on a degenerate axis to the midpoint, 0.5.
    degeneratex = iszero(Δx)
    degeneratey = iszero(Δy)

    invΔx = degeneratex ? zero(Float32) : inv(Δx)
    invΔy = degeneratey ? zero(Float32) : inv(Δy)

    # Prepare the normalization offsets (fx = invΔx * xi + cx).
    cx = degeneratex ? Float32(0.5) : -invΔx * xmin
    cy = degeneratey ? Float32(0.5) : -invΔy * ymin

    # Allocate vectors for the final result
    codes = MtlVector{TI}(undef, npart)
    order = similar(codes)

    # Encode all points without allocating normalized or quantized coordinates
    @metal threads=(ThreadsPerGroup,) groups=(cld(npart, ThreadsPerGroup),) Partia.LinearBoundingVolumeHierarchy._morton_encoding_kernel!(codes, (xcopy, ycopy), (invΔx, invΔy), (cx, cy))
    Metal.synchronize()

    # Construct structure
    enc = Partia.MortonEncoding{2, Float32, TI, MtlVector{Float32}, MtlVector{TI}}(order, codes, (xcopy, ycopy))

    # Sort by morton
    Partia.sort_by_morton!(enc, SortWorkSpace, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))
    return enc
end

"""
    MortonEncoding(points::NTuple{2,MtlVector{Float32}}, ::Val{TileSize}=Val(2048), ::Val{NThreadgroups}=Val(128), ::Val{ThreadsPerGroup}=Val(256);
                   CodeType=UInt64,
                   SortWorkSpace=OnesweepWorkspace(MtlVector{CodeType}))

Encode a set of 2D particle coordinates into Morton codes.

This overload accepts particle positions in a structure-of-arrays (SoA) layout,
where `points = (x, y)`. It forwards to
`MortonEncoding(x, y; CodeType=CodeType)`.

# Parameters
- `points :: NTuple{2,MtlVector{Float32}}`: Particle coordinates stored as `(x, y)`.
- `::Val{TileSize}`: Compile-time tile size used by the OneSweep radix sorter.
- `::Val{NThreadgroups}`: Number of threadgroups used by the radix sorter.
- `::Val{ThreadsPerGroup}`: Number of threads in each Metal threadgroup.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |
| `SortWorkSpace` | `OnesweepWorkspace{TI}` | `OnesweepWorkspace(MtlVector{CodeType})` | Reusable workspace for Morton-code sorting. |

# Returns
- A 2D `MortonEncoding` with codes, indices, and coordinates ordered by Morton code.
"""
function Partia.MortonEncoding(points :: NTuple{2, MtlVector{Float32}}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256);
    CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(MtlVector{CodeType})) where {TileSize, NThreadgroups, ThreadsPerGroup, TI <: Unsigned}
    x = points[1]; y = points[2]
    return Partia.MortonEncoding(x, y, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup); CodeType, SortWorkSpace)
end

######################################################################################

# Morton encoding constructors for particle spatial indexing with Metal.
#     by Wei-Shan Su,
#     July 13, 2026

######################################################################################
"""
    MortonEncoding(x::MtlVector{Float32}, y::MtlVector{Float32}, z::MtlVector{Float32}; CodeType=UInt64)

Encode a set of 3D particle coordinates into Morton codes without sorting.
Call `sort_by_morton!` before constructing a `LinearBVH`.

# Parameters
- `x, y, z :: MtlVector{Float32}`: Particle positions along each axis (floating-point).

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |

# Returns
- `MortonEncoding{3, Float32, TI, MtlVector{Float32}, MtlVector{TI}}`: Unsorted
  encoding whose codes and copied coordinates remain in input order.
"""
function Partia.MortonEncoding(x :: MtlVector{Float32}, y :: MtlVector{Float32}, z :: MtlVector{Float32}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned}
    # Verify length of input arrays
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    isempty(y) && throw(ArgumentError("coordinates must not be empty"))
    isempty(z) && throw(ArgumentError("coordinates must not be empty"))

    axes(x) == axes(y) == axes(z) || throw(DimensionMismatch("x, y, and z must have identical axes"))

    # Copy the coordinates to prevent modification
    xcopy = copy(x); ycopy = copy(y); zcopy = copy(z)
    npart = length(xcopy)

    # Use separate scalar reductions because large tuple-valued `extrema`
    # reductions can reset the Metal command buffer.
    bounds = Partia.LinearBoundingVolumeHierarchy._coordinate_bounds((xcopy, ycopy, zcopy))
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
    @metal threads=(256,) groups=(cld(npart, 256),) Partia.LinearBoundingVolumeHierarchy._morton_encoding_kernel!(codes, (xcopy, ycopy, zcopy), (invΔx, invΔy, invΔz), (cx, cy, cz))
    Metal.synchronize()

    # Construct structure
    enc = Partia.MortonEncoding{3, Float32, TI, MtlVector{Float32}, MtlVector{TI}}(order, codes, (xcopy, ycopy, zcopy))

    return enc
end

"""
    MortonEncoding(points::NTuple{3,MtlVector{Float32}}; CodeType=UInt64)

Encode a set of 3D particle coordinates into Morton codes.

This overload accepts particle positions in a structure-of-arrays (SoA) layout,
where `points = (x, y, z)`. It forwards to
`MortonEncoding(x, y, z; CodeType=CodeType)`.

# Parameters
- `points :: NTuple{3,MtlVector{Float32}}`: Particle coordinates stored as `(x, y, z)`.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |

# Returns
- An unsorted 3D `MortonEncoding`; call `sort_by_morton!` to arrange it in
  Morton order.
"""
function Partia.MortonEncoding(points :: NTuple{3, MtlVector{Float32}}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned}
    x = points[1]; y = points[2]; z = points[3]
    return Partia.MortonEncoding(x, y, z; CodeType)
end

"""
    MortonEncoding(x::MtlVector{Float32}, y::MtlVector{Float32}; CodeType=UInt64)

Encode a set of 2D particle coordinates into Morton codes without sorting.
Call `sort_by_morton!` before constructing a `LinearBVH`.

# Parameters
- `x, y :: MtlVector{Float32}`: Particle positions along each axis (floating-point).

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |

# Returns
- `MortonEncoding{2, Float32, TI, MtlVector{Float32}, MtlVector{TI}}`: Unsorted
  encoding whose codes and copied coordinates remain in input order.
"""
function Partia.MortonEncoding(x :: MtlVector{Float32}, y :: MtlVector{Float32}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned}
    # Verify length of input arrays
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    isempty(y) && throw(ArgumentError("coordinates must not be empty"))

    axes(x) == axes(y) || throw(DimensionMismatch("x and y must have identical axes"))

    # Copy the coordinates to prevent modification
    xcopy = copy(x); ycopy = copy(y)
    npart = length(xcopy)

    # Use separate scalar reductions because large tuple-valued `extrema`
    # reductions can reset the Metal command buffer.
    bounds = Partia.LinearBoundingVolumeHierarchy._coordinate_bounds((xcopy, ycopy))
    xmin, xmax = bounds[1]
    ymin, ymax = bounds[2]

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
    @metal threads=(256,) groups=(cld(npart, 256),) Partia.LinearBoundingVolumeHierarchy._morton_encoding_kernel!(codes, (xcopy, ycopy), (invΔx, invΔy), (cx, cy))
    Metal.synchronize()

    # Construct structure
    enc = Partia.MortonEncoding{2, Float32, TI, MtlVector{Float32}, MtlVector{TI}}(order, codes, (xcopy, ycopy))

    return enc
end

"""
    MortonEncoding(points::NTuple{2,MtlVector{Float32}}; CodeType=UInt64)

Encode a set of 2D particle coordinates into Morton codes.

This overload accepts particle positions in a structure-of-arrays (SoA) layout,
where `points = (x, y)`. It forwards to
`MortonEncoding(x, y; CodeType=CodeType)`.

# Parameters
- `points :: NTuple{2,MtlVector{Float32}}`: Particle coordinates stored as `(x, y)`.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |

# Returns
- An unsorted 2D `MortonEncoding`; call `sort_by_morton!` to arrange it in
  Morton order.
"""
function Partia.MortonEncoding(points :: NTuple{2, MtlVector{Float32}}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned}
    x = points[1]; y = points[2]
    return Partia.MortonEncoding(x, y; CodeType)
end

"""
    MortonEncoding!(x::MtlVector{Float32}, y::MtlVector{Float32}; CodeType=UInt64)
    MortonEncoding!(x::MtlVector{Float32}, y::MtlVector{Float32}, z::MtlVector{Float32}; CodeType=UInt64)

Construct a Metal Morton encoding without copying coordinate vectors. The
coordinates remain in input order until `sort_by_morton!` is called.

# Parameters
- `x`, `y`, `z`: Nonempty Metal coordinate vectors with identical axes.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned Morton-code type. |

# Returns
- `MortonEncoding`: Unsorted Metal encoding that aliases the supplied vectors.
"""
function Partia.MortonEncoding!(x :: MtlVector{Float32}, y :: MtlVector{Float32}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned}
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    axes(x) == axes(y) || throw(DimensionMismatch("x and y must have identical axes"))
    codes = MtlVector{TI}(undef, length(x))
    enc = Partia.MortonEncoding{2, Float32, TI, MtlVector{Float32}, MtlVector{TI}}(similar(codes), codes, (x, y))
    Partia.build!(enc, x, y)
    return enc
end

function Partia.MortonEncoding!(x :: MtlVector{Float32}, y :: MtlVector{Float32}, z :: MtlVector{Float32}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned}
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    axes(x) == axes(y) == axes(z) || throw(DimensionMismatch("x, y, and z must have identical axes"))
    codes = MtlVector{TI}(undef, length(x))
    enc = Partia.MortonEncoding{3, Float32, TI, MtlVector{Float32}, MtlVector{TI}}(similar(codes), codes, (x, y, z))
    Partia.build!(enc, x, y, z)
    return enc
end

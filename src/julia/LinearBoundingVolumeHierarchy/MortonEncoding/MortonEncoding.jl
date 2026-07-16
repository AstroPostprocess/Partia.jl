######################################################################################

# Morton encoding data structure and constructors for particle spatial indexing.
#     by Wei-Shan Su,
#     July 12, 2026

######################################################################################
################# Define structures #################
"""
    MortonEncoding{D, TF, TI, VF, VI}

Store Morton codes, their sorting permutation, and structure-of-arrays
coordinates for a two- or three-dimensional point set.

# Fields
- `order :: VI`: Permutation populated by `sort_by_morton!`; it maps data in
  input order into the stored Morton order.
- `codes :: VI`: Morton codes, initially in input order and sorted in place by
  `sort_by_morton!`.
- `coord :: NTuple{D, VF}`: Coordinate vectors kept in the same order as
  `codes`.
"""
struct MortonEncoding{D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}}
    order :: VI             # Order of corresponding particles
    codes :: VI             # Morton code
    coord :: NTuple{D, VF}  # Original data points
end

"""Return per-axis coordinate bounds using scalar reductions."""
@inline function _coordinate_bounds(coords :: NTuple{D, V}) where {D, V <: AbstractVector}
    return map(coord -> (minimum(coord), maximum(coord)), coords)
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
    MortonEncoding(x::Vector{T}, y::Vector{T}, z::Vector{T}; CodeType=UInt64)

Encode a set of 3D particle coordinates into Morton codes without sorting.
The copied coordinates and codes remain in input order. Call
`sort_by_morton!` before passing the result to `LinearBVH`.

# Parameters
- `x, y, z :: Vector{T}`: Particle positions along each axis (floating-point).

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |

# Returns
- `MortonEncoding{3, T, TI, Vector{T}, Vector{TI}}`: Unsorted encoding containing
  Morton codes and copied coordinates. `order` is populated by `sort_by_morton!`.
"""
function MortonEncoding(x :: Vector{T}, y :: Vector{T}, z :: Vector{T}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T <: AbstractFloat}
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

    return enc
end

"""
    MortonEncoding(coords::NTuple{3,Vector{T}}; CodeType=UInt64)

Encode a set of 3D particle coordinates into Morton codes.

This overload accepts particle positions in a structure-of-arrays (SoA) layout,
where `coords = (x, y, z)`. It forwards to
`MortonEncoding(x, y, z; CodeType=CodeType)`.

# Parameters
- `coords :: NTuple{3,Vector{T}}`: Particle coordinates stored as `(x, y, z)`.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |

# Returns
- An unsorted 3D `MortonEncoding`; call `sort_by_morton!` to populate its
  permutation and arrange its codes and coordinates in Morton order.
"""
function MortonEncoding(coords :: NTuple{3, Vector{T}}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T <: AbstractFloat}
    x = coords[1]; y = coords[2]; z = coords[3]
    return MortonEncoding(x, y, z; CodeType)
end

"""
    MortonEncoding(x::Vector{T}, y::Vector{T}; CodeType=UInt64)

Encode a set of 2D particle coordinates into Morton codes without sorting.
The copied coordinates and codes remain in input order. Call
`sort_by_morton!` before passing the result to `LinearBVH`.

# Parameters
- `x, y :: Vector{T}`: Particle positions along each axis (floating-point).

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |

# Returns
- `MortonEncoding{2, T, TI, Vector{T}, Vector{TI}}`: Unsorted encoding containing
  Morton codes and copied coordinates. `order` is populated by `sort_by_morton!`.
"""
function MortonEncoding(x :: Vector{T}, y :: Vector{T}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T <: AbstractFloat}
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

    return enc
end

"""
    MortonEncoding(coords::NTuple{2,Vector{T}}; CodeType=UInt64)

Encode a set of 2D particle coordinates into Morton codes.

This overload accepts particle positions in a structure-of-arrays (SoA) layout,
where `coords = (x, y)`. It forwards to
`MortonEncoding(x, y; CodeType=CodeType)`.

# Parameters
- `coords :: NTuple{2,Vector{T}}`: Particle coordinates stored as `(x, y)`.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`). |

# Returns
- An unsorted 2D `MortonEncoding`; call `sort_by_morton!` to populate its
  permutation and arrange its codes and coordinates in Morton order.
"""
function MortonEncoding(coords :: NTuple{2, Vector{T}}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T <: AbstractFloat}
    x = coords[1]; y = coords[2]
    return MortonEncoding(x, y; CodeType)
end

"""
    MortonEncoding!(x::Vector{TF}, y::Vector{TF}; CodeType=UInt64)
    MortonEncoding!(x::Vector{TF}, y::Vector{TF}, z::Vector{TF}; CodeType=UInt64)

Construct a Morton encoding without copying coordinate vectors. The supplied
`x`, `y`, and optional `z` vectors become `enc.coord` and remain in input order
until `sort_by_morton!` is called. Use `MortonEncoding` instead when inputs must
be preserved.

# Parameters
- `x`, `y`, `z`: Two- or three-dimensional coordinate vectors. All supplied
  vectors must be nonempty and have identical axes.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned integer type used for Morton codes. |

# Returns
- `MortonEncoding`: Unsorted encoding that aliases the supplied coordinate
  vectors. `order` is populated when `sort_by_morton!` is called.
"""
function MortonEncoding!(x :: Vector{TF}, y :: Vector{TF}; CodeType :: Type{TI} = UInt64) where {TF <: AbstractFloat, TI <: Unsigned}
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    axes(x) == axes(y) || throw(DimensionMismatch("x and y must have identical axes"))
    codes = Vector{TI}(undef, length(x))
    enc = MortonEncoding{2, TF, TI, Vector{TF}, Vector{TI}}(similar(codes), codes, (x, y))
    build!(enc, x, y)
    return enc
end

function MortonEncoding!(x :: Vector{TF}, y :: Vector{TF}, z :: Vector{TF}; CodeType :: Type{TI} = UInt64) where {TF <: AbstractFloat, TI <: Unsigned}
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    axes(x) == axes(y) == axes(z) || throw(DimensionMismatch("x, y, and z must have identical axes"))
    codes = Vector{TI}(undef, length(x))
    enc = MortonEncoding{3, TF, TI, Vector{TF}, Vector{TI}}(similar(codes), codes, (x, y, z))
    build!(enc, x, y, z)
    return enc
end

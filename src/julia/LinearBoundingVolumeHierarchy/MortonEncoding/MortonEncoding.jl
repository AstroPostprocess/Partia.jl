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

function Adapt.adapt_structure(to, x :: ME) where {D, ME <: MortonEncoding{D}}
    MortonEncoding(
        Adapt.adapt(to, x.order),
        Adapt.adapt(to, x.codes),
        ntuple(i -> Adapt.adapt(to, x.coord[i]), D)
    )
end

################# Encoding Morton code #################
"""
    MortonEncoding(x :: V, y :: V, z :: V; CodeType :: Type{TI}=UInt64)

Encode a set of 3D particle coordinates into Morton codes.

# Parameters
- `x, y, z :: AbstractVector{T}`: Particle positions along each axis (floating-point).
- `CodeType :: Type{TI}`: Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`).

# Returns
- `MortonEncoding{3, T, TI, V, typeof(order)}`: Encoding containing Morton codes,
  original particle indices, and copied coordinates, all ordered by Morton code.
"""
function MortonEncoding(x :: V, y :: V, z :: V; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T <: AbstractFloat, V <: AbstractVector{T}}
    # Verify length of input arrays
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    isempty(y) && throw(ArgumentError("coordinates must not be empty"))
    isempty(z) && throw(ArgumentError("coordinates must not be empty"))

    axes(x) == axes(y) == axes(z) || throw(DimensionMismatch("x, y, and z must have identical axes"))

    # Copy the coordinate to prevent modification
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

    invΔx = degeneratex ? zero(T) : inv(Δx)
    invΔy = degeneratey ? zero(T) : inv(Δy)
    invΔz = degeneratez ? zero(T) : inv(Δz)

    # prepare the offset of normalization (fx = invΔx * xi + (- invΔx * xmin))
    cx = degeneratex ? T(0.5) : -invΔx * xmin
    cy = degeneratey ? T(0.5) : -invΔy * ymin
    cz = degeneratez ? T(0.5) : -invΔz * zmin

    # Allocate vector for final result
    codes = Vector{TI}(undef, npart)
    order = similar(codes)

    # Go through all the points
    @inbounds @threads for i in eachindex(codes, xcopy, ycopy, zcopy)
        _morton_encoding_kernel!(codes, i, (xcopy, ycopy, zcopy), (invΔx, invΔy, invΔz), (cx, cy, cz))
    end

    # Construct structure
    enc = MortonEncoding{3, T, TI, V, typeof(order)}(order, codes, (xcopy, ycopy, zcopy))

    # Sort by morton
    sort_by_morton!(enc)
    return enc
end

"""
    MortonEncoding(points :: NTuple{3,V}; CodeType :: Type{TI}=UInt64) where {TI <: Unsigned, T <: AbstractFloat, V <: AbstractVector{T}}

Encode a set of 3D particle coordinates into Morton codes.

This overload accepts particle positions in a structure-of-arrays (SoA) layout,
where `points = (x, y, z)`. It forwards to
`MortonEncoding(x, y, z; CodeType=CodeType)`.

# Parameters
- `points :: NTuple{3,V}`: Particle coordinates stored as `(x, y, z)`.
- `CodeType :: Type{TI}`: Unsigned integer type used for Morton encoding
  (`UInt32` or `UInt64`).

# Returns
- A 3D `MortonEncoding` with codes, indices, and coordinates ordered by Morton code.
"""
function MortonEncoding(points :: NTuple{3, V}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T <: AbstractFloat, V <: AbstractVector{T}}
    x = points[1]; y = points[2]; z = points[3]
    return MortonEncoding(x, y, z, CodeType = CodeType)
end

"""
    MortonEncoding(x :: V, y :: V; CodeType :: Type{TI}=UInt64)

Encode a set of 2D particle coordinates into Morton codes.

# Parameters
- `x, y :: AbstractVector{T}`: Particle positions along each axis (floating-point).
- `CodeType :: Type{TI}`: Unsigned integer type used for Morton encoding
  (`UInt32` or `UInt64`).

# Returns
- `MortonEncoding{2, T, TI, V, typeof(order)}`: Encoding containing Morton codes,
  original particle indices, and copied coordinates, all ordered by Morton code.
"""
function MortonEncoding(x :: V, y :: V; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T <: AbstractFloat, V <: AbstractVector{T}}
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

    enc = MortonEncoding{2, T, TI, V, typeof(order)}(order, codes, (xcopy, ycopy))
    sort_by_morton!(enc)
    return enc
end

"""
    MortonEncoding(points :: NTuple{2,V}; CodeType :: Type{TI}=UInt64) where {TI <: Unsigned, T <: AbstractFloat, V <: AbstractVector{T}}

Encode a set of 2D particle coordinates into Morton codes.

This overload accepts particle positions in a structure-of-arrays (SoA) layout,
where `points = (x, y)`. It forwards to
`MortonEncoding(x, y; CodeType=CodeType)`.

# Parameters
- `points :: NTuple{2,V}`: Particle coordinates stored as `(x, y)`.
- `CodeType :: Type{TI}`: Unsigned integer type used for Morton encoding
  (`UInt32` or `UInt64`).

# Returns
- A 2D `MortonEncoding` with codes, indices, and coordinates ordered by Morton code.
"""
function MortonEncoding(points :: NTuple{2, V}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T <: AbstractFloat, V <: AbstractVector{T}}
    x = points[1]; y = points[2]
    return MortonEncoding(x, y, CodeType = CodeType)
end

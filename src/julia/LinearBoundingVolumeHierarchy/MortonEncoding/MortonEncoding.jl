######################################################################################

# Morton encoding data structure and constructors for particle spatial indexing.
#     by Wei-Shan Su,
#     May 4, 2026

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
- `MortonEncoding{3, T, TI, V, typeof(order)}`: Struct containing Morton codes, particle order, and coordinates, ordered by Morton codes
"""
function MortonEncoding(x :: V, y :: V, z :: V; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T <: AbstractFloat, V <: AbstractVector{T}}
    xcopy = copy(x); ycopy = copy(y); zcopy = copy(z)
    ix, iy, iz = _quantize_coords(xcopy, ycopy, zcopy, CodeType=CodeType)
    codes, order  = _encode_morton_code3D(ix, iy, iz)
    enc = MortonEncoding{3, T, TI, V, typeof(order)}(order, codes, (xcopy, ycopy, zcopy))
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
- `points :: NTuple{3,V}` :
  Particle coordinates in 3D, stored as `(x, y, z)`, where each vector has length `N`.
- `CodeType :: Type{TI}=UInt64` :
  Unsigned integer type used for Morton encoding (typically `UInt32` or `UInt64`).

# Returns
The same return value as `MortonEncoding(x, y, z; CodeType=CodeType)`.
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
- `CodeType :: Type{TI}`: Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`).

# Returns
- `MortonEncoding{2, T, TI, V, typeof(order)}`: Struct containing Morton codes, particle order, and coordinates, ordered by Morton codes
"""
function MortonEncoding(x :: V, y :: V; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T <: AbstractFloat, V <: AbstractVector{T}}
    xcopy = copy(x); ycopy = copy(y)
    ix, iy = _quantize_coords(xcopy, ycopy, CodeType=CodeType)
    codes, order  = _encode_morton_code2D(ix, iy)
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
- `points :: NTuple{2,V}` :
  Particle coordinates in 2D, stored as `(x, y)`, where each vector has length `N`.
- `CodeType :: Type{TI}=UInt64` :
  Unsigned integer type used for Morton encoding (typically `UInt32` or `UInt64`).

# Returns
The same return value as `MortonEncoding(x, y; CodeType=CodeType)`.
"""
function MortonEncoding(points :: NTuple{2, V}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T <: AbstractFloat, V <: AbstractVector{T}}
    x = points[1]; y = points[2]
    return MortonEncoding(x, y, CodeType = CodeType)
end


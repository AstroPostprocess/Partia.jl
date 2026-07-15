######################################################################################

# Morton encoding constructors for particle spatial indexing with CUDA.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
"""
    MortonEncoding(points::NTuple{D,CuVector{TF}}; CodeType=UInt64)
    MortonEncoding(x::CuVector{TF}, y::CuVector{TF}; CodeType=UInt64)
    MortonEncoding(x::CuVector{TF}, y::CuVector{TF}, z::CuVector{TF}; CodeType=UInt64)

Encode two- or three-dimensional CUDA coordinates into Morton codes without
sorting. `TF` may be any CUDA-supported subtype of `AbstractFloat`, and `D`
must be either 2 or 3. Call `sort_by_morton!` before constructing a `LinearBVH`.

# Parameters
- `points`: Structure-of-arrays coordinates `(x, y)` or `(x, y, z)`.
- `x`, `y`, `z`: Coordinate-wise convenience arguments for `points`.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned Morton-code type. |

# Returns
- `MortonEncoding`: Unsorted CUDA encoding whose codes and copied coordinates
  remain in input order. `sort_by_morton!` populates `order`.
"""
function Partia.MortonEncoding(points :: NTuple{D, CuVector{TF}}; CodeType :: Type{TI} = UInt64) where {D, TF <: AbstractFloat, TI <: Unsigned}
    # Validate the structure-of-arrays input before launching a GPU kernel.
    D in (2, 3) || throw(ArgumentError("Morton encoding only supports two or three dimensions"))
    all(!isempty, points) || throw(ArgumentError("coordinates must not be empty"))
    all(axes(p) == axes(points[1]) for p in points) || throw(DimensionMismatch("coordinates must have identical axes"))

    # Allocate reusable encoding storage, then populate it through the in-place path.
    coord = map(similar, points)
    codes = CuVector{TI}(undef, length(points[1]))
    order = similar(codes)
    enc = Partia.MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}(order, codes, coord)
    Partia.build!(enc, points)
    return enc
end

function Partia.MortonEncoding(x :: CuVector{TF}, y :: CuVector{TF}, args...; kwargs...) where {TF <: AbstractFloat}
    return Partia.MortonEncoding((x, y), args...; kwargs...)
end

function Partia.MortonEncoding(x :: CuVector{TF}, y :: CuVector{TF}, z :: CuVector{TF}, args...; kwargs...) where {TF <: AbstractFloat}
    return Partia.MortonEncoding((x, y, z), args...; kwargs...)
end

"""
    MortonEncoding!(x::CuVector{TF}, y::CuVector{TF}; CodeType=UInt64)
    MortonEncoding!(x::CuVector{TF}, y::CuVector{TF}, z::CuVector{TF}; CodeType=UInt64)

Construct a CUDA Morton encoding without copying coordinate vectors. The
coordinates remain in input order until `sort_by_morton!` is called.

# Parameters
- `x`, `y`, `z`: Nonempty CUDA coordinate vectors with identical axes.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned Morton-code type. |

# Returns
- `MortonEncoding`: Unsorted CUDA encoding that aliases the supplied vectors.
"""
function Partia.MortonEncoding!(x :: CuVector{TF}, y :: CuVector{TF}; CodeType :: Type{TI} = UInt64) where {TF <: AbstractFloat, TI <: Unsigned}
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    axes(x) == axes(y) || throw(DimensionMismatch("x and y must have identical axes"))
    codes = CuVector{TI}(undef, length(x))
    enc = Partia.MortonEncoding{2, TF, TI, CuVector{TF}, CuVector{TI}}(similar(codes), codes, (x, y))
    Partia.build!(enc, x, y)
    return enc
end

function Partia.MortonEncoding!(x :: CuVector{TF}, y :: CuVector{TF}, z :: CuVector{TF}; CodeType :: Type{TI} = UInt64) where {TF <: AbstractFloat, TI <: Unsigned}
    isempty(x) && throw(ArgumentError("coordinates must not be empty"))
    axes(x) == axes(y) == axes(z) || throw(DimensionMismatch("x, y, and z must have identical axes"))
    codes = CuVector{TI}(undef, length(x))
    enc = Partia.MortonEncoding{3, TF, TI, CuVector{TF}, CuVector{TI}}(similar(codes), codes, (x, y, z))
    Partia.build!(enc, x, y, z)
    return enc
end

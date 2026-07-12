######################################################################################

# InterpolationSmoothingVolumeInput.jl
#     by Wei-Shan Su
#     October 31, 2025
# Definition of the smoothing-volume interpolation input structure.
# This input mirrors `InterpolationInput`, but omits the density column and uses
# the PHANTOM smoothing-length relation
#
#     m_b / rho_b = h_b^D / hfact^D
#
# inside the interpolation kernels. It is intended for specialized paths where
# this relation is the defining volume element.

######################################################################################
"""
    InterpolationSmoothingVolumeInput{D, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel, NCOLUMN}

Immutable SPH input container for smoothing-volume interpolation queries.

This struct stores the same coordinate, mass, smoothing-length, and quantity
columns as `InterpolationInput`, but does not store a density column. Instead,
the interpolation kernels use the smoothing-volume relation
`m_b / rho_b = h_b^D / hfact^D`, where `hfact` is the smoothing-length ratio
used by PHANTOM.

# Type Parameters
- `D`: Spatial dimension.
- `T`: Floating-point type (e.g. `Float32` or `Float64`).
- `V`: An `AbstractVector{T}`, the vector type used throughout.
- `K`: Type of SPH kernel used. Must be a concrete `AbstractSPHKernel`.
- `NCOLUMN`: Number of scalar fields stored in `quant`.

# Fields
- `Npart :: Int64`: Number of active (valid) particles within the batch.
- `hfact :: T`: Smoothing-length ratio used in `m_b / rho_b = h_b^D / hfact^D`.
- `smoothed_kernel :: K`: SPH kernel function instance.
- `coord :: NTuple{D,V}`: Particle coordinates, e.g. `(x, y, z)` for `D == 3`.
- `m :: V`: Particle masses. Stored for metadata and particle ordering, but the
  smoothing-volume interpolation weights are derived from `h` and `hfact`.
- `h :: V`: Particle smoothing lengths.
- `quant :: NTuple{NCOLUMN,V}`: Tuple of per-field scalar data arrays.
"""
struct InterpolationSmoothingVolumeInput{D, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel, NCOLUMN} <: AbstractInterpolationInput{D, T, V, K, NCOLUMN}
    Npart :: Int64
    hfact :: T
    smoothed_kernel :: K
    coord :: NTuple{D, V}
    m :: V
    h :: V
    quant :: NTuple{NCOLUMN, V}
end

function Adapt.adapt_structure(to, x :: InterpolationSmoothingVolumeInput{D, T, V, K, NCOLUMN}) where {D, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel, NCOLUMN}
    InterpolationSmoothingVolumeInput(
        x.Npart,
        x.hfact,
        Adapt.adapt(to, x.smoothed_kernel),
        ntuple(i -> Adapt.adapt(to, x.coord[i]), Val(D)),
        Adapt.adapt(to, x.m),
        Adapt.adapt(to, x.h),
        ntuple(i -> Adapt.adapt(to, x.quant[i]), Val(NCOLUMN)),
    )
end

"""
    InterpolationSmoothingVolumeInput(hfact :: T, coord :: NTuple{D,V}, m :: V, h :: V, quant :: NTuple{NCOLUMN,V}; smoothed_kernel :: Type{K} = M5_spline) where {D, NCOLUMN, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel}

Construct a smoothing-volume interpolation input from materialized particle
columns.

Unlike `InterpolationInput`, this constructor does not take a density array.
The corresponding interpolation kernels assume the relation
`m_b / rho_b = h_b^D / hfact^D`.

# Parameters
- `hfact :: T`: Smoothing-length ratio used to define the volume element.
- `coord :: NTuple{D,V}`: Coordinate tuple, such as `(x, y, z)`.
- `m :: V`: Particle masses.
- `h :: V`: Particle smoothing lengths.
- `quant :: NTuple{NCOLUMN,V}`: Tuple of scalar field columns.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `smoothed_kernel` | `Type{K}` | `M5_spline` | Kernel type used to construct the stored kernel instance. |

# Returns
- `InterpolationSmoothingVolumeInput{D,T,V,K,NCOLUMN}`: Smoothing-volume input
  with validated column lengths.
"""
function InterpolationSmoothingVolumeInput(hfact :: T, coord :: NTuple{D, V}, m :: V, h :: V, quant :: NTuple{NCOLUMN, V}; smoothed_kernel :: Type{K} = M5_spline) where {D, NCOLUMN, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel}
    Npart = length(m)
    @inbounds for d in 1:D
        length(coord[d]) == Npart || throw(
            DimensionMismatch("coord[$d] length $(length(coord[d])) != Nparticles $Npart"),
        )
    end
    length(h) == Npart || throw(DimensionMismatch("h length $(length(h)) != Nparticles $Npart"))
    @inbounds for j in 1:NCOLUMN
        length(quant[j]) == Npart || throw(
            DimensionMismatch("quant[$j] length $(length(quant[j])) != Nparticles $Npart"),
        )
    end

    return InterpolationSmoothingVolumeInput{D, T, V, K, NCOLUMN}(
        Npart,
        hfact,
        smoothed_kernel(),
        coord,
        m,
        h,
        quant,
    )
end

# Basic constructors
"""
    InterpolationSmoothingVolumeInput(hfact :: T, x :: V, y :: V, m :: V, h :: V, quant :: NTuple{NCOLUMN,V}; smoothed_kernel :: Type{K} = M5_spline) where {NCOLUMN, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel}

Construct a 2D smoothing-volume interpolation input from separate coordinate
vectors.

# Parameters
- `hfact :: T`: Smoothing-length ratio used to define the volume element.
- `x :: V`: Particle x-coordinates.
- `y :: V`: Particle y-coordinates.
- `m :: V`: Particle masses.
- `h :: V`: Particle smoothing lengths.
- `quant :: NTuple{NCOLUMN,V}`: Tuple of scalar field columns.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `smoothed_kernel` | `Type{K}` | `M5_spline` | Kernel type used to construct the stored kernel instance. |

# Returns
- `InterpolationSmoothingVolumeInput{2,T,V,K,NCOLUMN}`: 2D smoothing-volume
  interpolation input.
"""
@inline function InterpolationSmoothingVolumeInput(hfact :: T, x :: V, y :: V, m :: V, h :: V, quant :: NTuple{NCOLUMN, V}; smoothed_kernel :: Type{K} = M5_spline) where {NCOLUMN, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel}
    return InterpolationSmoothingVolumeInput(hfact, (x, y), m, h, quant; smoothed_kernel = smoothed_kernel)
end

"""
    InterpolationSmoothingVolumeInput(hfact :: T, x :: V, y :: V, z :: V, m :: V, h :: V, quant :: NTuple{NCOLUMN,V}; smoothed_kernel :: Type{K} = M5_spline) where {NCOLUMN, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel}

Construct a 3D smoothing-volume interpolation input from separate coordinate
vectors.

# Parameters
- `hfact :: T`: Smoothing-length ratio used to define the volume element.
- `x :: V`: Particle x-coordinates.
- `y :: V`: Particle y-coordinates.
- `z :: V`: Particle z-coordinates.
- `m :: V`: Particle masses.
- `h :: V`: Particle smoothing lengths.
- `quant :: NTuple{NCOLUMN,V}`: Tuple of scalar field columns.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `smoothed_kernel` | `Type{K}` | `M5_spline` | Kernel type used to construct the stored kernel instance. |

# Returns
- `InterpolationSmoothingVolumeInput{3,T,V,K,NCOLUMN}`: 3D smoothing-volume
  interpolation input.
"""
@inline function InterpolationSmoothingVolumeInput(hfact :: T, x :: V, y :: V, z :: V, m :: V, h :: V, quant :: NTuple{NCOLUMN, V}; smoothed_kernel :: Type{K} = M5_spline) where {NCOLUMN, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel}
    return InterpolationSmoothingVolumeInput(hfact, (x, y, z), m, h, quant; smoothed_kernel = smoothed_kernel)
end

# Check the "Valid" length of data for each fields
function Base.checkbounds(input :: InterpolationSmoothingVolumeInput)
    N = input.Npart
    @assert N isa Integer && N >= 0 "Invalid Npart: $N"

    @inbounds for d in 1:spatial_dimension(input)
        @assert N <= length(input.coord[d]) "coord[$d] is shorter than Npart ($N)"
    end

    @assert N <= length(input.m) "m is shorter than Npart ($N)"
    @assert N <= length(input.h) "h is shorter than Npart ($N)"

    @inbounds for (k, v) in enumerate(input.quant)
        @assert N <= length(v) "quant[$k] is shorter than Npart ($N)"
    end
    return true
end

# Input helper for LBVH
## 3D path
"""
    LinearBVH!(input :: InterpolationSmoothingVolumeInput{3}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned}

Build a 3D `LinearBVH` for a smoothing-volume interpolation input.

The input arrays are permuted in-place into Morton leaf order, matching the
behavior of `LinearBVH!(input :: InterpolationInput{3})`. The mass,
smoothing-length, and quantity columns are reordered together with the
coordinates.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned Morton-code integer type. |

# Returns
- `LinearBVH{3}`: Linear bounding volume hierarchy with leaf scales taken from
  `input.h`.
"""
function LinearBVH!(input :: InterpolationSmoothingVolumeInput{3}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned}
    x = get_xcoord(input)
    y = get_ycoord(input)
    z = get_zcoord(input)

    enc = MortonEncoding(x, y, z, CodeType = CodeType)
    order = enc.order

    Base.permute!(x, order)
    Base.permute!(y, order)
    Base.permute!(z, order)
    Base.permute!(input.m, order)
    Base.permute!(input.h, order)
    for column in input.quant
        Base.permute!(column, order)
    end

    return LinearBVH(enc, input.h)
end

## 2D path
"""
    LinearBVH!(input :: InterpolationSmoothingVolumeInput{2}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned}

Build a 2D `LinearBVH` for a smoothing-volume interpolation input.

The input arrays are permuted in-place into Morton leaf order. The mass,
smoothing-length, and quantity columns are reordered together with the
coordinates.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned Morton-code integer type. |

# Returns
- `LinearBVH{2}`: Linear bounding volume hierarchy with leaf scales taken from
  `input.h`.
"""
function LinearBVH!(input :: InterpolationSmoothingVolumeInput{2}; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned}
    x = get_xcoord(input)
    y = get_ycoord(input)

    enc = MortonEncoding(x, y, CodeType = CodeType)
    order = enc.order

    Base.permute!(x, order)
    Base.permute!(y, order)
    Base.permute!(input.m, order)
    Base.permute!(input.h, order)
    for column in input.quant
        Base.permute!(column, order)
    end

    return LinearBVH(enc, input.h)
end

"""
    matches_lbvh_leaf_order(input :: InterpolationSmoothingVolumeInput{D}, lbvh :: LinearBVH{D}) where {D}

Check whether a smoothing-volume interpolation input is already arranged in the
same Morton-reordered leaf order as a given `LinearBVH`.

This function is intended as a lightweight consistency check when an externally
supplied `LinearBVH` is reused together with an
`InterpolationSmoothingVolumeInput`. It compares the reordered spatial arrays
stored in `input` against the leaf data stored in `lbvh`.

# Parameters
- `input :: InterpolationSmoothingVolumeInput{D}`: Input whose current particle
  ordering is to be validated.
- `lbvh :: LinearBVH{D}`: Linear bounding volume hierarchy whose leaf ordering is
  treated as the reference ordering.

# Returns
- `Bool`: `true` when `input.coord` matches the leaf section of
  `lbvh.aabb.min` and `input.h` matches the leaf section of `lbvh.scale`.
"""
@inline function matches_lbvh_leaf_order(input :: InterpolationSmoothingVolumeInput{D}, lbvh :: LinearBVH{D}) :: Bool where {D}
    leaf_nodes = lbvh.nleaf:(2 * lbvh.nleaf - 1)
    all(input.coord[d] == @view(lbvh.aabb.min[d][leaf_nodes]) for d in 1:D) &&
        input.h == @view(lbvh.scale[leaf_nodes])
end

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
@inline function InterpolationSmoothingVolumeInput(hfact :: T, x :: V, y :: V, m :: V, h :: V, quant :: NTuple{NCOLUMN, V}; smoothed_kernel :: Type{K} = M5_spline) where {NCOLUMN, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel}
    return InterpolationSmoothingVolumeInput(hfact, (x, y), m, h, quant; smoothed_kernel = smoothed_kernel)
end

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

    brt = BinaryRadixTree(enc)
    return LinearBVH(enc, brt, BoxScale(input.h, true))
end

## 2D path
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

    brt = BinaryRadixTree(enc)
    return LinearBVH(enc, brt, BoxScale(input.h, true))
end

@inline function matches_lbvh_leaf_order(input :: InterpolationSmoothingVolumeInput{D}, lbvh :: LinearBVH{D}) :: Bool where {D}
    all(input.coord[d] == lbvh.leaf_coor[d] for d in 1:D) && (input.h == lbvh.leaf_scale)
end

# Input helper for LBVH
## 3D path
function Partia.LinearBVH!(input :: InterpolationInput{3, Float32, MtlVector{Float32}}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256);
    CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(MtlVector{CodeType})) where {TileSize, NThreadgroups, ThreadsPerGroup, TI <: Unsigned}
    x = get_xcoord(input)
    y = get_ycoord(input)
    z = get_zcoord(input)

    enc = Partia.MortonEncoding(x, y, z; CodeType)
    Partia.sort_by_morton!(enc, SortWorkSpace, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))
    order = enc.order

    Base.permute!(x, order)
    Base.permute!(y, order)
    Base.permute!(z, order)
    Base.permute!(input.m, order)
    Base.permute!(input.h, order)
    Base.permute!(input.ρ, order)
    for column in input.quant
        Base.permute!(column, order)
    end

    return Partia.LinearBVH(enc, input.h, Val(ThreadsPerGroup))
end

## 2D path
function Partia.LinearBVH!(input :: InterpolationInput{2, Float32, MtlVector{Float32}}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256);
    CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(MtlVector{CodeType})) where {TileSize, NThreadgroups, ThreadsPerGroup, TI <: Unsigned}
    x = get_xcoord(input)
    y = get_ycoord(input)

    enc = Partia.MortonEncoding(x, y; CodeType)
    Partia.sort_by_morton!(enc, SortWorkSpace, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))
    order = enc.order

    Base.permute!(x, order)
    Base.permute!(y, order)
    Base.permute!(input.m, order)
    Base.permute!(input.h, order)
    Base.permute!(input.ρ, order)
    for column in input.quant
        Base.permute!(column, order)
    end

    return Partia.LinearBVH(enc, input.h, Val(ThreadsPerGroup))
end

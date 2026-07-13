# Input helper for LBVH
## 3D path
function Partia.LinearBVH!(input :: InterpolationInput{3, TF, CuVector{TF}}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256);
    CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(CuVector{CodeType})) where {TF <: AbstractFloat, TileSize, NBlocks, ThreadsPerBlock, TI <: Unsigned}
    x = get_xcoord(input)
    y = get_ycoord(input)
    z = get_zcoord(input)

    enc = Partia.MortonEncoding(x, y, z, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock); CodeType, SortWorkSpace)
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

    return Partia.LinearBVH(enc, input.h, Val(NBlocks), Val(ThreadsPerBlock))
end

## 2D path
function Partia.LinearBVH!(input :: InterpolationInput{2, TF, CuVector{TF}}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256);
    CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(CuVector{CodeType})) where {TF <: AbstractFloat, TileSize, NBlocks, ThreadsPerBlock, TI <: Unsigned}
    x = get_xcoord(input)
    y = get_ycoord(input)

    enc = Partia.MortonEncoding(x, y, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock); CodeType, SortWorkSpace)
    order = enc.order

    Base.permute!(x, order)
    Base.permute!(y, order)
    Base.permute!(input.m, order)
    Base.permute!(input.h, order)
    Base.permute!(input.ρ, order)
    for column in input.quant
        Base.permute!(column, order)
    end

    return Partia.LinearBVH(enc, input.h, Val(NBlocks), Val(ThreadsPerBlock))
end

######################################################################################

# CUDA LBVH setup for density-based interpolation input.

######################################################################################
"""
    LinearBVH!(input::InterpolationInput{D,TF,CuVector{TF}},
               ::Val{TileSize}=Val(4096),
               ::Val{NBlocks}=Val(256),
               ::Val{ThreadsPerBlock}=Val(256);
               CodeType=UInt64,
               SortWorkSpace=OnesweepWorkspace(CuVector{CodeType}))

Build a CUDA LBVH for an interpolation input. Coordinates, mass, smoothing
length, density, and every quantity column are permuted in-place into the same
Morton leaf order before the hierarchy is constructed.

`D` may be 2 or 3 and `TF` may be any CUDA-supported `AbstractFloat` subtype.
The returned hierarchy uses `input.h` as its per-leaf scale.
"""
function Partia.LinearBVH!(input :: InterpolationInput{D, TF, CuVector{TF}}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256);
    CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(CuVector{CodeType})) where {D, TF <: AbstractFloat, TileSize, NBlocks, ThreadsPerBlock, TI <: Unsigned}
    # MortonEncoding copies and sorts coordinates, while order is subsequently
    # applied to every field owned by input.
    coords = Partia.get_coord(input)
    enc = Partia.MortonEncoding(coords, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock); CodeType, SortWorkSpace)
    order = enc.order
    # Keep all particle attributes aligned with the LBVH leaf order.
    foreach(v -> Base.permute!(v, order), coords)
    Base.permute!(input.m, order)
    Base.permute!(input.h, order)
    Base.permute!(input.ρ, order)
    foreach(v -> Base.permute!(v, order), input.quant)
    # Coordinates represent point leaves; smoothing length is the search scale.
    return Partia.LinearBVH(enc, input.h, Val(NBlocks), Val(ThreadsPerBlock))
end

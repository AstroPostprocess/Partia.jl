######################################################################################

# CUDA LBVH setup for smoothing-volume interpolation input.

######################################################################################
"""
    LinearBVH!(input::InterpolationSmoothingVolumeInput{D,TF,CuVector{TF}},
               ::Val{TileSize}=Val(4096),
               ::Val{NBlocks}=Val(256),
               ::Val{ThreadsPerBlock}=Val(256);
               CodeType=UInt64,
               SortWorkSpace=OnesweepWorkspace(CuVector{CodeType}))

Build a CUDA LBVH for a smoothing-volume interpolation input. Coordinates,
mass, smoothing length, and quantity columns are permuted in-place into Morton
leaf order. The returned hierarchy uses `input.h` as its per-leaf scale.

# Parameters
- `TileSize`: OneSweep radix-sort tile size; defaults to 4096.
- `NBlocks`: CUDA block count; defaults to 256.
- `ThreadsPerBlock`: CUDA threads per block; defaults to 256.
"""
function Partia.LinearBVH!(input :: InterpolationSmoothingVolumeInput{D, TF, CuVector{TF}}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256);
    CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(CuVector{CodeType})) where {D, TF <: AbstractFloat, TileSize, NBlocks, ThreadsPerBlock, TI <: Unsigned}
    # Generate one shared Morton permutation for every particle attribute.
    coords = Partia.get_coord(input)
    enc = Partia.MortonEncoding(coords, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock); CodeType, SortWorkSpace)
    order = enc.order
    # Keep the structure-of-arrays fields aligned with the LBVH leaves.
    foreach(v -> Base.permute!(v, order), coords)
    Base.permute!(input.m, order)
    Base.permute!(input.h, order)
    foreach(v -> Base.permute!(v, order), input.quant)
    return Partia.LinearBVH(enc, input.h, Val(NBlocks), Val(ThreadsPerBlock))
end

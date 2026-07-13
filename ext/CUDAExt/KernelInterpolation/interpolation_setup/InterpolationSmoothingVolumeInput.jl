# Input helper for LBVH
## 3D path
"""
    LinearBVH!(input :: InterpolationSmoothingVolumeInput{3}, ::Val{TileSize}=Val(4096), ::Val{NBlocks}=Val(256), ::Val{ThreadsPerBlock}=Val(256);
                CodeType=UInt64,
                SortWorkSpace=OnesweepWorkspace(CuVector{CodeType}))

Build a 3D `LinearBVH` for a smoothing-volume interpolation input.

The input arrays are permuted in-place into Morton leaf order, matching the
behavior of `LinearBVH!(input :: InterpolationInput{3})`. The mass,
smoothing-length, and quantity columns are reordered together with the
coordinates.

# Parameters
- `input :: InterpolationSmoothingVolumeInput{3}`: Smoothing-volume interpolation input stored on CUDA.
- `::Val{TileSize}`: Compile-time tile size used by the OneSweep radix sorter.
- `::Val{NBlocks}`: Number of CUDA blocks used by the radix sorter.
- `::Val{ThreadsPerBlock}`: Number of threads used by the CUDA kernels.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned Morton-code integer type. |
| `SortWorkSpace` | `OnesweepWorkspace{TI}` | `OnesweepWorkspace(CuVector{CodeType})` | Reusable workspace for Morton-code sorting. |

# Returns
- `LinearBVH{3}`: Linear bounding volume hierarchy with leaf scales taken from
  `input.h`.
"""
function Partia.LinearBVH!(input :: InterpolationSmoothingVolumeInput{3, TF, CuVector{TF}}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256);
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
    for column in input.quant
        Base.permute!(column, order)
    end

    return Partia.LinearBVH(enc, input.h, Val(NBlocks), Val(ThreadsPerBlock))
end

## 2D path
"""
    LinearBVH!(input :: InterpolationSmoothingVolumeInput{2}, ::Val{TileSize}=Val(4096), ::Val{NBlocks}=Val(256), ::Val{ThreadsPerBlock}=Val(256);
                CodeType=UInt64,
                SortWorkSpace=OnesweepWorkspace(CuVector{CodeType}))

Build a 2D `LinearBVH` for a smoothing-volume interpolation input.

The input arrays are permuted in-place into Morton leaf order. The mass,
smoothing-length, and quantity columns are reordered together with the
coordinates.

# Parameters
- `input :: InterpolationSmoothingVolumeInput{2}`: Smoothing-volume interpolation input stored on CUDA.
- `::Val{TileSize}`: Compile-time tile size used by the OneSweep radix sorter.
- `::Val{NBlocks}`: Number of CUDA blocks used by the radix sorter.
- `::Val{ThreadsPerBlock}`: Number of threads used by the CUDA kernels.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `CodeType` | `Type{TI}` | `UInt64` | Unsigned Morton-code integer type. |
| `SortWorkSpace` | `OnesweepWorkspace{TI}` | `OnesweepWorkspace(CuVector{CodeType})` | Reusable workspace for Morton-code sorting. |

# Returns
- `LinearBVH{2}`: Linear bounding volume hierarchy with leaf scales taken from
  `input.h`.
"""
function Partia.LinearBVH!(input :: InterpolationSmoothingVolumeInput{2, TF, CuVector{TF}}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256);
    CodeType :: Type{TI} = UInt64, SortWorkSpace :: OnesweepWorkspace{TI} = OnesweepWorkspace(CuVector{CodeType})) where {TF <: AbstractFloat, TileSize, NBlocks, ThreadsPerBlock, TI <: Unsigned}
    x = get_xcoord(input)
    y = get_ycoord(input)

    enc = Partia.MortonEncoding(x, y, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock); CodeType, SortWorkSpace)
    order = enc.order

    Base.permute!(x, order)
    Base.permute!(y, order)
    Base.permute!(input.m, order)
    Base.permute!(input.h, order)
    for column in input.quant
        Base.permute!(column, order)
    end

    return Partia.LinearBVH(enc, input.h, Val(NBlocks), Val(ThreadsPerBlock))
end

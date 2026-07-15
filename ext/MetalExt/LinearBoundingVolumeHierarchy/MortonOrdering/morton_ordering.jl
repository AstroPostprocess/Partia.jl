"""
    sort_by_morton!(enc, workspace,
                    ::Val{TileSize}=Val(2048),
                    ::Val{NThreadgroups}=Val(128),
                    ::Val{ThreadsPerGroup}=Val(256))
    sort_by_morton!(enc,
                    ::Val{TileSize}=Val(2048),
                    ::Val{NThreadgroups}=Val(128),
                    ::Val{ThreadsPerGroup}=Val(256))

Sort Metal Morton codes with OneSweep, store the permutation in `enc.order`,
and apply it to every coordinate vector in `enc.coord`.

# Parameters
- `enc`: Metal Morton encoding to sort in place.
- `workspace`: Reusable Metal OneSweep workspace compatible with `enc.codes`.
- `TileSize`: Compile-time OneSweep tile size. Defaults to 2048.
- `NThreadgroups`: Number of Metal threadgroups. Defaults to 128.
- `ThreadsPerGroup`: Number of threads per threadgroup. Defaults to 256.

# Returns
- Sorting permutation, also copied into `enc.order` and applied to `enc.coord`.
"""
function Partia.sort_by_morton!(enc :: MortonEncoding{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, ws :: OnesweepWorkspace{TI, CodeV, OffsetV}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {D, TileSize, NThreadgroups, ThreadsPerGroup, TI <: Unsigned, CodeV <: MtlVector{TI}, OffsetV <: MtlVector{UInt32}}
    p = radix_sortperm!(enc.codes, ws, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))

    copyto!(enc.order, p)

    for dir in enc.coord
        Base.permute!(dir, p)
    end

    return p
end

@inline function Partia.sort_by_morton!(enc :: MortonEncoding{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {D, TileSize, NThreadgroups, ThreadsPerGroup, TI <: Unsigned}
    ws = OnesweepWorkspace(typeof(enc.codes))
    return Partia.sort_by_morton!(enc, ws, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))
end

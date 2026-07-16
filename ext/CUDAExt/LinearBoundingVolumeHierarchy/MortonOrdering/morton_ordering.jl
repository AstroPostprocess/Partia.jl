"""
    sort_by_morton!(enc, workspace,
                    ::Val{TileSize}=Val(4096),
                    ::Val{NBlocks}=Val(256),
                    ::Val{ThreadsPerBlock}=Val(256))
    sort_by_morton!(enc,
                    ::Val{TileSize}=Val(4096),
                    ::Val{NBlocks}=Val(256),
                    ::Val{ThreadsPerBlock}=Val(256))

Sort CUDA Morton codes with OneSweep, store the permutation in `enc.order`,
and apply it to every coordinate vector in `enc.coord`.

# Parameters
- `enc`: CUDA Morton encoding to sort in place.
- `workspace`: Reusable CUDA OneSweep workspace compatible with `enc.codes`.
- `TileSize`: Compile-time OneSweep tile size. Defaults to 4096.
- `NBlocks`: Number of CUDA blocks. Defaults to 256.
- `ThreadsPerBlock`: Number of threads per CUDA block. Defaults to 256.

# Returns
- Sorting permutation, also copied into `enc.order` and applied to `enc.coord`.
"""
function Partia.sort_by_morton!(enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, ws :: OnesweepWorkspace{TI, CodeV, OffsetV}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, TileSize, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned, CodeV <: CuVector{TI}, OffsetV <: CuVector{UInt32}}
    p = onesweep_sortperm!(enc.codes, ws, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock))

    # Preserve the permutation for callers that need to reorder associated data.
    copyto!(enc.order, p)

    # Keep every coordinate axis aligned with the sorted Morton codes.
    for dir in enc.coord
        Base.permute!(dir, p)
    end

    return p
end


@inline function Partia.sort_by_morton!(enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, TileSize, NBlocks, ThreadsPerBlock, TF <: AbstractFloat, TI <: Unsigned}
    ws = OnesweepWorkspace(typeof(enc.codes))
    return Partia.sort_by_morton!(enc, ws, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock))
end

"""
    sort_by_morton!(enc, workspace,
                    ::Val{TileSize}=Val(4096),
                    ::Val{NBlocks}=Val(256),
                    ::Val{ThreadsPerBlock}=Val(256))

Sort CUDA Morton codes with OneSweep, store the permutation in `enc.order`,
and apply it to every coordinate vector in `enc.coord`.
"""
function Partia.sort_by_morton!(enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, ws :: OnesweepWorkspace{TI, CuVector{TI}, OffsetV}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, TF <: AbstractFloat, TI <: Unsigned, OffsetV <: CuVector{UInt32}, TileSize, NBlocks, ThreadsPerBlock}
    p = onesweep_sortperm!(enc.codes, ws, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock))

    # Preserve the permutation for callers that need to reorder associated data.
    copyto!(enc.order, p)

    # Keep every coordinate axis aligned with the sorted Morton codes.
    for dir in enc.coord
        Base.permute!(dir, p)
    end

    return p
end


@inline function Partia.sort_by_morton!(enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, :: Val{TileSize} = Val(4096), :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, TF <: AbstractFloat, TI <: Unsigned, TileSize, NBlocks, ThreadsPerBlock}
    ws = OnesweepWorkspace(typeof(enc.codes))
    return Partia.sort_by_morton!(enc, ws, Val(TileSize), Val(NBlocks), Val(ThreadsPerBlock))
end

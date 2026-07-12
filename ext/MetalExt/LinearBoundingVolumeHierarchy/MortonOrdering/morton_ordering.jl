function Partia.sort_by_morton!(enc :: MortonEncoding{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, ws :: OnesweepWorkspace{TI, MtlVector{TI}, MtlVector{UInt32}}, :: Val{TileSize}) where {D, TI <: Unsigned, TileSize}
    p = onesweep_sortperm!(enc.codes, ws, Val(TileSize))

    copyto!(enc.order, p)

    for dir in enc.coord
        Base.permute!(dir, p)
    end

    return p
end


@inline function sort_by_morton!(enc :: MortonEncoding{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}}) where {D, TI <: Unsigned}
    ws = OnesweepWorkspace(typeof(enc.codes))
    return sort_by_morton!(enc, ws, Val(4096))
end

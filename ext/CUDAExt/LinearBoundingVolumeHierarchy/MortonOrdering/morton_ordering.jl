function Partia.sort_by_morton!(enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}, ws :: OnesweepWorkspace{TI, CuVector{TI}, CuVector{UInt32}}, :: Val{TileSize}) where {D, TF <: AbstractFloat, TI <: Unsigned, TileSize}
    p = onesweep_sortperm!(enc.codes, ws, Val(TileSize))

    copyto!(enc.order, p)

    for dir in enc.coord
        Base.permute!(dir, p)
    end

    return p
end


@inline function sort_by_morton!(enc :: MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}) where {D, TF <: AbstractFloat, TI <: Unsigned}
    ws = OnesweepWorkspace(typeof(enc.codes))
    return sort_by_morton!(enc, ws, Val(4096))
end

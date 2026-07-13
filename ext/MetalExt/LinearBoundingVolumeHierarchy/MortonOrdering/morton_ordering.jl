function Partia.sort_by_morton!(enc :: MortonEncoding{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}}, ws :: OnesweepWorkspace{TI, CodeV, OffsetV}, :: Val{TileSize} = Val(2048), :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {D, TileSize, NThreadgroups, ThreadsPerGroup, TI <: Unsigned, CodeV <: MtlVector{TI}, OffsetV <: MtlVector{UInt32}}
    p = onesweep_sortperm!(enc.codes, ws, Val(TileSize), Val(NThreadgroups), Val(ThreadsPerGroup))

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

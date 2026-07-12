######################################################################################

# Morton-order permutation helpers.
#     by Wei-Shan Su,
#     July 12, 2026

######################################################################################
"""
    sort_by_morton!(enc::MortonEncoding)
    sort_by_morton!(enc::MortonEncoding, ws::OnesweepWorkspace,
                    ::Val{TileSize})

Sort a `Vector`-backed encoding in-place by ascending Morton code with the
OneSweep radix sorter. The same permutation is applied to `enc.order` and every
coordinate vector in `enc.coord`.

# Parameters
- `enc`: `Vector`-backed Morton encoding to mutate.
- `ws`: OneSweep workspace whose key-vector type matches `enc.codes`. Reuse it
  across calls to avoid repeated workspace allocation.
- `::Val{TileSize}`: Compile-time OneSweep tile size.

# Returns
The 1-based sorting permutation.
"""
@inline function sort_by_morton!(enc :: MortonEncoding{D, TF, TI, Vector{TF}, Vector{TI}}, ws :: OnesweepWorkspace{TI, Vector{TI}, Vector{UInt32}}, :: Val{TileSize}) where {D, TF <: AbstractFloat, TI <: Unsigned, TileSize}
    p = onesweep_sortperm!(enc.codes, ws, Val(TileSize))

    copyto!(enc.order, p)

    for dir in enc.coord
        Base.permute!(dir, p)
    end

    return p
end

@inline function sort_by_morton!(enc :: MortonEncoding{D, TF, TI, Vector{TF}, Vector{TI}}) where {D, TF <: AbstractFloat, TI <: Unsigned}
    ws = OnesweepWorkspace(typeof(enc.codes))
    return sort_by_morton!(enc, ws, Val(4096))
end

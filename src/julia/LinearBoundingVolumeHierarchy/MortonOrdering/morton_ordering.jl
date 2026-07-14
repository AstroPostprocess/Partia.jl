######################################################################################

# Morton-order permutation helpers.
#     by Wei-Shan Su,
#     July 12, 2026

######################################################################################
"""
    sort_by_morton!(
        enc::MortonEncoding{D, TF, TI, Vector{TF}, Vector{TI}},
        ::Val{TileSize} = Val(8192),
    ) where {D, TileSize, TF <: AbstractFloat, TI <: Unsigned}

    sort_by_morton!(
        enc::MortonEncoding{D, TF, TI, Vector{TF}, Vector{TI}},
        ws::OnesweepWorkspace{TI, Vector{TI}, Vector{UInt32}},
        ::Val{TileSize} = Val(8192),
    ) where {D, TileSize, TF <: AbstractFloat, TI <: Unsigned}

Sort a `Vector`-backed Morton encoding in ascending Morton-code order using the
OneSweep radix sorter. The resulting permutation is stored in `enc.order` and
applied in place to every coordinate vector in `enc.coord`.

# Parameters
- `enc`: `Vector`-backed Morton encoding to sort in place.
- `ws`: OneSweep workspace compatible with `enc.codes`. Reuse the workspace
  across calls to avoid repeated allocation.
- `::Val{TileSize}`: Compile-time OneSweep tile size. Defaults to `Val(8192)`.

# Returns
The 1-based sorting permutation.
"""
@inline function sort_by_morton!(enc :: MortonEncoding{D, TF, TI, Vector{TF}, Vector{TI}}, ws :: OnesweepWorkspace{TI, Vector{TI}, Vector{UInt32}}, :: Val{TileSize} = Val(8192)) where {D, TileSize, TF <: AbstractFloat, TI <: Unsigned}
    p = onesweep_sortperm!(enc.codes, ws, Val(TileSize))

    copyto!(enc.order, p)

    for dir in enc.coord
        Base.permute!(dir, p)
    end

    return p
end

@inline function sort_by_morton!(enc :: MortonEncoding{D, TF, TI, Vector{TF}, Vector{TI}}, :: Val{TileSize} = Val(8192)) where {D, TileSize, TF <: AbstractFloat, TI <: Unsigned}
    ws = OnesweepWorkspace(typeof(enc.codes))
    return sort_by_morton!(enc, ws, Val(TileSize))
end

######################################################################################

# Morton-order permutation helpers.
#     by Wei-Shan Su,
#     May 4, 2026

######################################################################################
"""
    sort_by_morton!(enc :: MortonEncoding)

Sort particles by Morton code in-place.

# Parameters
- `enc :: MortonEncoding`: The encoding struct to be sorted.

# Returns
- `p :: Vector{Int}`: The permutation indices used for sorting.
"""
@inline function sort_by_morton!(enc :: MortonEncoding)
    p = sortperm(enc.codes; alg=QuickSort)
    @inbounds for i in eachindex(enc.order)
        enc.order[i] = i
    end
    Base.permute!(enc.codes, p)
    Base.permute!(enc.order, p)
    for dir in enc.coord
        Base.permute!(dir, p)
    end
    @inbounds for i in 2:length(enc.codes)
        if enc.codes[i] <= enc.codes[i - 1]
            enc.codes[i] = enc.codes[i-1] + one(eltype(enc.codes))
        end
    end
    return p
end

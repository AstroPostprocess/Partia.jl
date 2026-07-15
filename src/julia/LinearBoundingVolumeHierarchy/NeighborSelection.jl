"""
    NeighborSelection{TI, VI}

Store a reusable neighbor-index pool, the valid prefix length, and an optional
nearest-neighbor index.

# Fields
- `pool :: VI`: Reusable index storage.
- `count :: TI`: Number of valid entries at the beginning of `pool`.
- `nearest :: TI`: Selected nearest-neighbor index.
"""
struct NeighborSelection{TI <: Integer, VI <: AbstractVector{TI}}
    pool :: VI
    count :: TI
    nearest :: TI
end

function Adapt.adapt_structure(to, x :: NeighborSelection)
    NeighborSelection(
        Adapt.adapt(to, x.pool),
        x.count,
        x.nearest
    )
end

Base.length(result :: NeighborSelection) = result.count

"""
    valid_indices(result::NeighborSelection)

Return a view of the valid prefix `result.pool[1:result.count]`.

# Parameters
- `result`: Neighbor-selection result.

# Returns
- `SubArray`: Non-copying view containing the selected indices.
"""
@inline function valid_indices(result :: NeighborSelection)
    return @view result.pool[1:result.count]
end

"""
    nearest_index(result::NeighborSelection)

Return the nearest-neighbor index stored in `result`.

# Parameters
- `result`: Neighbor-selection result.

# Returns
- `Integer`: Stored nearest-neighbor index.
"""
@inline function nearest_index(result :: NeighborSelection)
    return result.nearest
end

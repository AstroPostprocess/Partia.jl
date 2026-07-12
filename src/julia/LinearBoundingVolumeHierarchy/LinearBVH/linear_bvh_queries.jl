"""
    LBVH_probe_neighbors(LBVH, point, radius)

Find leaf bounding boxes whose distance from `point` does not exceed `radius`.
Traversal uses the unified node layout and returns particle-order leaf indices,
not unified node IDs.

# Parameters
- `LBVH :: LinearBVH{D,T}`: Hierarchy containing unified AABBs, scales, and
  stackless `left`/`escape` topology.
- `point :: NTuple{D,T}`: Query point in D-dimensional space.
- `radius :: T`: Search radius.

# Returns
A 3-tuple `(count, closest_idx, closest_dist2)`:
- `count :: Int`: Number of leaf AABBs intersecting the query sphere.
- `closest_idx :: Int`: Index of the closest intersecting leaf (0 if none).
- `closest_dist2 :: T`: Minimum squared distance to an intersecting leaf
  (`typemax(T)` if none).
"""
@inline function LBVH_probe_neighbors(LBVH :: LinearBVH{D, T}, point :: NTuple{D, T}, radius :: T) where {D, T <: AbstractFloat}
    # Initialize
    r2 = radius * radius
    count = 0
    closest_idx = 0
    closest_dist2 = typemax(T)

    # Traversal
    leaf_idx :: Int = zero(Int)
    point_to_leaf_aabb_d2 :: T = zero(T)

    @LBVH_gather_point_traversal LBVH point r2 leaf_idx point_to_leaf_aabb_d2 begin
        count += 1
        if point_to_leaf_aabb_d2 < closest_dist2
            closest_dist2 = point_to_leaf_aabb_d2
            closest_idx = leaf_idx
        end
    end

    return count, closest_idx, closest_dist2
end

"""
    LBVH_find_nearest(LBVH, point)

Find the leaf AABB nearest to a query point in a `LinearBVH`.

The search uses stackless depth-first traversal over `LBVH.left` and
`LBVH.escape`. Unified node IDs are partitioned as `1:nleaf-1` for internal
nodes and `nleaf:2nleaf-1` for leaves. Node `1` is visited first.

- For an internal node, if the point-to-AABB squared distance exceeds the current
  `best_dist2`, the whole subtree is pruned by jumping to `escape[node]`.
- For a leaf node, the leaf is considered only when its point-to-AABB squared
  distance is within the current bound, and `best_idx/best_dist2` are updated if
  it improves the best.

# Parameters
- `LBVH :: LinearBVH{D,T}`:
  Linear BVH containing `nleaf`, `left`, `escape`, and unified `aabb`/`scale`
  arrays of length `2nleaf-1`.

- `point :: NTuple{D,T}`:
  Query point in D-dimensional space. Values are assumed finite.

# Returns
A 2-tuple `(best_idx, best_dist2)`:
- `best_idx :: Int`:
  Leaf index (1-based) of the closest leaf AABB. Returns `0` only if the BVH
  contains no leaves (should not happen if `nleaf ≥ 1`).
- `best_dist2 :: T`:
  Squared distance from `point` to the closest leaf AABB.
"""
@inline function LBVH_find_nearest(LBVH :: LinearBVH{D,T}, point :: NTuple{D,T}) where {D,T <: AbstractFloat}
    # Initial best distance set to +∞
    best_idx = 0
    best_dist2 = typemax(T)

    # Traversal
    leaf_idx :: Int = zero(Int)
    point_to_leaf_aabb_d2 :: T = zero(T)

    @LBVH_gather_point_traversal LBVH point best_dist2 leaf_idx point_to_leaf_aabb_d2 begin
        if point_to_leaf_aabb_d2 < best_dist2
            best_dist2 = point_to_leaf_aabb_d2
            best_idx = leaf_idx
        end
    end
    return best_idx, best_dist2
end

"""
    LBVH_find_nearest_h(LBVH :: LinearBVH{D,T}, point :: NTuple{D,T}) where {D,T <: AbstractFloat}

Return the scale stored for the leaf AABB nearest to `point` in a
`LinearBVH`.

Both internal pruning and leaf acceptance use point-to-AABB squared distance.
For a returned leaf index `i`, its unified node ID is `nleaf - 1 + i`, and the
result is read from `LBVH.scale[nleaf - 1 + i]`.

# Parameters
- `LBVH :: LinearBVH{D,T}`
  Bounding volume hierarchy with unified node AABBs and per-node scales.
- `point :: NTuple{D,T}`
  Query point in the same coordinate space as the particles.

# Returns
- `h :: T`
  Smoothing length of the nearest particle (leaf) to `point`.
"""
@inline function LBVH_find_nearest_h(LBVH :: LinearBVH{D,T}, point :: NTuple{D,T}) where {D,T <: AbstractFloat}
    # Initial best distance set to +∞
    best_idx = 0
    best_dist2 = typemax(T)

    # Smoothed radius
    unified_scale = LBVH.scale

    # Traversal
    leaf_idx :: Int = zero(Int)
    point_to_leaf_aabb_d2 :: T = zero(T)

    @LBVH_gather_point_traversal LBVH point best_dist2 leaf_idx point_to_leaf_aabb_d2 begin
        if point_to_leaf_aabb_d2 < best_dist2
            best_dist2 = point_to_leaf_aabb_d2
            best_idx = leaf_idx
        end
    end

    best_idx == 0 && return T(NaN)   # 或 return (0, typemax(T))
    best_h = unified_scale[(LBVH.nleaf - 1) + best_idx]
    return best_h
end

"""
    LBVH_query!(pool, LBVH, point, radius)

Collect leaf indices whose leaf AABBs intersect a spherical query region centered
at `point` with radius `radius`, using a **stackless depth-first traversal** of a
`LinearBVH`.

Traversal is driven directly by `LBVH.left` and `LBVH.escape`. Nodes are visited
in DFS preorder starting at unified node `1`; accepted leaves are written into
`pool` using leaf indices in `1:nleaf`. Leaf AABBs are tested against `radius^2`.

# Parameters
- `pool :: AbstractVector{Int}`:
  Output buffer that receives accepted leaf indices (1-based). The function writes
  into `pool[1:count]`; the remaining entries are untouched.

- `LBVH :: LinearBVH{D,T}`:
  Linear BVH containing unified `aabb`/`scale` storage and stackless topology.

- `point :: NTuple{D,T}`:
  Query point in D-dimensional space.

- `radius`:
  Query radius. In the `radius :: T` method, `r2 = radius*radius` is used directly.
  In the `radius :: S` method, `radius` is promoted to `T`.

# Returns
- `NeighborSelection`:
  Handle/view describing the valid prefix of `pool` and the closest leaf among the
  accepted set (by squared distance). The neighbor count is `count`, and the
  closest leaf index is `closest_idx` (0 if no leaf is accepted).

# Notes
- The traversal order is determined by `left`/`escape` (DFS preorder in unified node
  ID space). This is not the same mechanism as the older parent-pointer traversal.
- Correctness assumes the `left`/`escape` tables and leaf/internal ID mapping
  (`is_leaf_id`, `leaf_index`, `internal_index`) are consistent with the LBVH build.
"""
@inline function LBVH_query!(pool :: VI, LBVH :: LinearBVH{D, T},
                                       point :: NTuple{D, T},
                                       radius :: T) where {D, T <: AbstractFloat, VI <: AbstractVector{Int}}
    # Initializing
    r2 = radius * radius
    count = 0
    closest_idx = zero(eltype(pool))
    closest_dist2 = typemax(T)

    # Traversal
    leaf_idx :: Int = zero(Int)
    point_to_leaf_aabb_d2 :: T = zero(T)

    @LBVH_gather_point_traversal LBVH point r2 leaf_idx point_to_leaf_aabb_d2 begin
        count += 1
        @inbounds pool[count] = leaf_idx
        if point_to_leaf_aabb_d2 < closest_dist2
          closest_dist2 = point_to_leaf_aabb_d2
          closest_idx = leaf_idx
        end
    end
    return NeighborSelection(pool, count, closest_idx)
end

@inline function LBVH_query!(pool :: VI, LBVH :: LinearBVH{D, T},
                                       point :: NTuple{D, T},
                                       radius :: S) where {D, T <: AbstractFloat, S <: AbstractFloat, VI <: AbstractVector{Int}}
    return LBVH_query!(pool, LBVH, point, T(radius))
end

"""
    @LBVH_gather_point_traversal(LBVH, reference_point, radius2, leafsym, d2sym, leaf_hit)

Stackless DFS traversal over a `LinearBVH` using its `left` and `escape` links.
Traversal starts at unified node `1`; internal nodes occupy `1:nleaf-1` and
leaf nodes occupy `nleaf:2nleaf-1`.

This is the **gather** variant: the pruning radius is given by the `radius2` input.

# Arguments
- `LBVH`: Hierarchy with unified `aabb` storage and stackless topology.
- `reference_point`:
  `NTuple{D, T}` query position.
- `radius2`:
  Squared query radius (same float type as distance computations).
- `leafsym`:
  Caller-scope symbol receiving the accepted leaf index in `1:nleaf`. A leaf's
  storage is accessed with its unified node ID, not this output index.
- `d2sym`:
  Caller-scope symbol that receives the squared point-to-leaf distance.
- `leaf_hit`:
  An expression executed on each accepted leaf after assigning `leafsym` and
  `d2sym` in caller scope.
"""
macro LBVH_gather_point_traversal(LBVH, reference_point, radius2, leafsym, d2sym, leaf_hit)
    # hygiene: private/local variable declaration
    node_min_       = gensym(:node_min)
    node_max_       = gensym(:node_max)
    left_           = gensym(:left)
    escape_         = gensym(:escape)
    nleaf_          = gensym(:nleaf)
    node_           = gensym(:node)
    node_idx_       = gensym(:node_idx)
    leaf_idx_       = gensym(:leaf_idx)
    d2_             = gensym(:d2)
    d2_node_        = gensym(:d2_node)

    # hygiene: avoid capturing user locals
    LBVH_       = esc(LBVH)
    rp_         = esc(reference_point)
    r2_         = esc(radius2)
    leafsym_    = esc(leafsym)
    d2sym_      = esc(d2sym)

    # the user-provided code must run in caller scope
    hit_  = esc(leaf_hit)

    quote
        # LBVH data
        ## AABB
        $node_min_    = $LBVH_.aabb.min
        $node_max_    = $LBVH_.aabb.max
        ## Topology
        $left_        = $LBVH_.left
        $escape_      = $LBVH_.escape

        ## Other information
        $nleaf_       = nleaf($LBVH_)

        $node_ = one(Int32)
        while !iszero($node_)
            # Leaf: process then jump by escape
            if is_leaf_id($node_, $nleaf_)
                $leaf_idx_ = leaf_index($node_, $nleaf_)
                $d2_ = _squared_distance_point_aabb($rp_, $node_min_, $node_max_, Int($node_))
                if $d2_ <= $r2_
                    $leafsym_ = $leaf_idx_
                    $d2sym_   = $d2_
                    $hit_
                end
                @inbounds $node_ = $escape_[Int($node_)]
                continue
            end

            # Internal: AABB reject => prune subtree
            $node_idx_ = internal_index($node_)
            $d2_node_ = _squared_distance_point_aabb($rp_, $node_min_, $node_max_, $node_idx_)
            if $d2_node_ > $r2_
                @inbounds $node_ = $escape_[Int($node_)]
                continue
            end

            # Internal: descend to left child (DFS preorder)
            @inbounds $node_ = $left_[$node_idx_]
        end
        nothing
    end
end

"""
    @LBVH_scatter_point_traversal(LBVH, reference_point, Kvalid, leafsym, d2sym, hbsym, leaf_hit)

Stackless DFS traversal over unified `LinearBVH` nodes using `left` and `escape`.

This is the **scatter** variant: the pruning radius is **node-dependent**.
For every node, the acceptance radius is `Kvalid * LBVH.scale[node_id]`.
Leaf callbacks still receive a leaf index in `1:nleaf`.

# Arguments
- `LBVH`: Hierarchy holding unified `aabb`/`scale` arrays and `left`/`escape` topology.
- `reference_point`: query point used in AABB distance tests.
- `Kvalid`: scalar multiplier converting smoothing length to search radius.
- `leafsym`:
  Caller-scope symbol that receives the accepted leaf index in `1:nleaf`.
- `d2sym`:
  Caller-scope symbol that receives the squared point-to-leaf distance.
- `hbsym`:
  Caller-scope symbol that receives the smoothing length associated with the
  accepted leaf.
- `leaf_hit`:
   An expression executed on each accepted leaf after assigning `leafsym`,
   `d2sym`, and `hbsym` in caller scope.
"""
macro LBVH_scatter_point_traversal(LBVH, reference_point, Kvalid, leafsym, d2sym, hbsym, leaf_hit)
    # hygiene: private/local variable declaration
    node_min_       = gensym(:node_min)
    node_max_       = gensym(:node_max)
    left_           = gensym(:left)
    escape_         = gensym(:escape)
    nleaf_          = gensym(:nleaf)
    node_           = gensym(:node)
    node_idx_       = gensym(:node_idx)
    leaf_idx_       = gensym(:leaf_idx)
    d2_             = gensym(:d2)
    d2_node_        = gensym(:d2_node)
    r_              = gensym(:r)
    r2_             = gensym(:r2)
    scale_          = gensym(:scale)
    hb_             = gensym(:hb)

    # hygiene: avoid capturing user locals
    LBVH_       = esc(LBVH)
    Kvalid_     = esc(Kvalid)
    rp_         = esc(reference_point)
    leafsym_    = esc(leafsym)
    d2sym_      = esc(d2sym)
    hbsym_      = esc(hbsym)

    # the user-provided code must run in caller scope
    hit_  = esc(leaf_hit)

    quote
        # LBVH data
        ## AABB
        $node_min_    = $LBVH_.aabb.min
        $node_max_    = $LBVH_.aabb.max
        ## Topology
        $left_        = $LBVH_.left
        $escape_      = $LBVH_.escape

        ## Other information
        $nleaf_       = nleaf($LBVH_)
        $scale_       = $LBVH_.scale

        $node_ = one(Int32)
        while !iszero($node_)
            # Leaf: process then jump by escape
            if is_leaf_id($node_, $nleaf_)
                $leaf_idx_ = leaf_index($node_, $nleaf_)
                $hb_    = $scale_[Int($node_)]
                $r_     = $Kvalid_ * $hb_
                $r2_    = $r_ * $r_
                $d2_ = _squared_distance_point_aabb($rp_, $node_min_, $node_max_, Int($node_))
                if $d2_ <= $r2_
                    $leafsym_ = $leaf_idx_
                    $d2sym_   = $d2_
                    $hbsym_   = $hb_
                    $hit_
                end
                @inbounds $node_ = $escape_[Int($node_)]
                continue
            end

            # Internal: AABB reject => prune subtree
            $node_idx_ = internal_index($node_)
            $hb_    = $scale_[$node_idx_]
            $r_     = $Kvalid_ * $hb_
            $r2_    = $r_ * $r_
            $d2_node_ = _squared_distance_point_aabb($rp_, $node_min_, $node_max_, $node_idx_)
            if $d2_node_ > $r2_
                @inbounds $node_ = $escape_[Int($node_)]
                continue
            end

            # Internal: descend to left child (DFS preorder)
            @inbounds $node_ = $left_[$node_idx_]
        end
        nothing
    end
end

"""
    @LBVH_gather_line_traversal(LBVH, line_origin, line_direction, radius2, leafsym, d2sym, leaf_hit)

Stackless DFS traversal over unified `LinearBVH` nodes using `left` and
`escape`, with no explicit stack or recursion.

This is the **gather** line-traversal variant: the pruning radius is given
directly by the input `radius2`. Internal nodes are pruned using a
conservative lower bound on squared line-to-AABB distance. Leaf nodes are also
represented by AABBs and use their unified node ID for the same test.

# Parameters
- `LBVH`: Hierarchy with unified AABBs and stackless `left`/`escape` topology.
- `line_origin`:
  `NTuple{D,T}` giving the origin of the query line.
- `line_direction`:
  `NTuple{D,T}` giving the direction of the query line. This direction is
  assumed to be a unit vector.
- `radius2`:
  Squared query radius used for both internal-node pruning and leaf
  acceptance.
- `leafsym`:
  Caller-scope symbol that receives the accepted leaf index in `1:nleaf`.
- `d2sym`:
  Caller-scope symbol that receives the squared distance from the query line
  to the accepted leaf primitive.
- `leaf_hit`:
  An expression executed on each accepted leaf after assigning `leafsym` and
  `d2sym` in caller scope.
"""
macro LBVH_gather_line_traversal(LBVH, line_origin, line_direction, radius2, leafsym, d2sym, leaf_hit)
    # Internal and leaf nodes share unified AABB storage.
    # hygiene: private/local variable declaration
    node_min_       = gensym(:node_min)
    node_max_       = gensym(:node_max)
    left_           = gensym(:left)
    escape_         = gensym(:escape)
    nleaf_          = gensym(:nleaf)
    node_           = gensym(:node)
    node_idx_       = gensym(:node_idx)
    leaf_idx_       = gensym(:leaf_idx)
    d2_             = gensym(:d2)
    d2_node_        = gensym(:d2_node)

    # hygiene: avoid capturing user locals
    LBVH_       = esc(LBVH)
    origin_     = esc(line_origin)
    direction_  = esc(line_direction)
    r2_         = esc(radius2)
    leafsym_    = esc(leafsym)
    d2sym_      = esc(d2sym)

    # the user-provided code must run in caller scope
    hit_  = esc(leaf_hit)

    quote
        # LBVH data
        ## AABB
        $node_min_    = $LBVH_.aabb.min
        $node_max_    = $LBVH_.aabb.max
        ## Topology
        $left_        = $LBVH_.left
        $escape_      = $LBVH_.escape

        ## Other information
        $nleaf_       = nleaf($LBVH_)

        $node_ = one(Int32)
        while !iszero($node_)
            # Leaf: process then jump by escape
            if is_leaf_id($node_, $nleaf_)
                $leaf_idx_ = leaf_index($node_, $nleaf_)
                $d2_ = _squared_distance_line_aabb_lower_bound($origin_, $direction_, $node_min_, $node_max_, Int($node_))
                if $d2_ <= $r2_
                    $leafsym_ = $leaf_idx_
                    $d2sym_   = $d2_
                    $hit_
                end
                @inbounds $node_ = $escape_[Int($node_)]
                continue
            end

            # Internal: AABB reject => prune subtree
            $node_idx_ = internal_index($node_)
            $d2_node_ = _squared_distance_line_aabb_lower_bound($origin_, $direction_, $node_min_, $node_max_, $node_idx_)
            if $d2_node_ > $r2_
                @inbounds $node_ = $escape_[Int($node_)]
                continue
            end

            # Internal: descend to left child (DFS preorder)
            @inbounds $node_ = $left_[$node_idx_]
        end
        nothing
    end
end

"""
    @LBVH_scatter_line_traversal(LBVH, line_origin, line_direction, Kvalid, leafsym, d2sym, hbsym, leaf_hit)

Stackless DFS traversal over unified `LinearBVH` nodes using `left` and
`escape`, with no explicit stack or recursion.

This is the **scatter** line-traversal variant: the acceptance radius is
node-dependent. For each leaf, the acceptance radius is

    r = Kvalid * LBVH.scale[leaf_node_id]

and for each internal node, subtree pruning uses

    r = Kvalid * LBVH.scale[node_id]

Both leaf acceptance and internal pruning use the conservative squared
line-to-AABB lower bound over unified AABB storage.

# Parameters
- `LBVH`: Hierarchy holding unified `aabb`/`scale` arrays and `left`/`escape` topology.
- `line_origin`:
  `NTuple{D,T}` giving the origin of the query line.
- `line_direction`:
  `NTuple{D,T}` giving the direction of the query line. This direction is
  assumed to be a unit vector.
- `Kvalid`:
  Scalar multiplier converting smoothing length to search radius.
- `leafsym`:
  Caller-scope symbol that receives the accepted leaf index in `1:nleaf`.
- `d2sym`:
  Caller-scope symbol that receives the squared distance from the query line
  to the accepted leaf primitive.
- `hbsym`:
  Caller-scope symbol that receives the smoothing length associated with the
  accepted leaf.
- `leaf_hit`:
  An expression executed on each accepted leaf after assigning `leafsym`,
  `d2sym`, and `hbsym` in caller scope.
"""
macro LBVH_scatter_line_traversal(LBVH, line_origin, line_direction, Kvalid, leafsym, d2sym, hbsym, leaf_hit)
    # Internal and leaf nodes share unified AABB and scale storage.
    # hygiene: private/local variable declaration
    node_min_       = gensym(:node_min)
    node_max_       = gensym(:node_max)
    left_           = gensym(:left)
    escape_         = gensym(:escape)
    nleaf_          = gensym(:nleaf)
    node_           = gensym(:node)
    node_idx_       = gensym(:node_idx)
    leaf_idx_       = gensym(:leaf_idx)
    d2_             = gensym(:d2)
    d2_node_        = gensym(:d2_node)
    r_              = gensym(:r)
    r2_             = gensym(:r2)
    scale_          = gensym(:scale)
    hb_             = gensym(:hb)

    # hygiene: avoid capturing user locals
    LBVH_       = esc(LBVH)
    Kvalid_     = esc(Kvalid)
    origin_     = esc(line_origin)
    direction_  = esc(line_direction)
    leafsym_    = esc(leafsym)
    d2sym_      = esc(d2sym)
    hbsym_      = esc(hbsym)

    # the user-provided code must run in caller scope
    hit_  = esc(leaf_hit)

    quote
        # LBVH data
        ## AABB
        $node_min_    = $LBVH_.aabb.min
        $node_max_    = $LBVH_.aabb.max
        ## Topology
        $left_        = $LBVH_.left
        $escape_      = $LBVH_.escape

        ## Other information
        $nleaf_       = nleaf($LBVH_)
        $scale_       = $LBVH_.scale

        $node_ = one(Int32)
        while !iszero($node_)
            # Leaf: process then jump by escape
            if is_leaf_id($node_, $nleaf_)
                $leaf_idx_ = leaf_index($node_, $nleaf_)
                $hb_    = $scale_[Int($node_)]
                $r_     = $Kvalid_ * $hb_
                $r2_    = $r_ * $r_
                $d2_ = _squared_distance_line_aabb_lower_bound($origin_, $direction_, $node_min_, $node_max_, Int($node_))
                if $d2_ <= $r2_
                    $leafsym_ = $leaf_idx_
                    $d2sym_   = $d2_
                    $hbsym_   = $hb_
                    $hit_
                end
                @inbounds $node_ = $escape_[Int($node_)]
                continue
            end

            # Internal: AABB reject => prune subtree
            $node_idx_ = internal_index($node_)
            $hb_    = $scale_[$node_idx_]
            $r_     = $Kvalid_ * $hb_
            $r2_    = $r_ * $r_
            $d2_node_ = _squared_distance_line_aabb_lower_bound($origin_, $direction_, $node_min_, $node_max_, $node_idx_)
            if $d2_node_ > $r2_
                @inbounds $node_ = $escape_[Int($node_)]
                continue
            end

            # Internal: descend to left child (DFS preorder)
            @inbounds $node_ = $left_[$node_idx_]
        end
        nothing
    end
end

@inline function _squared_distance_point_aabb(point :: NTuple{D,TF}, aabb_min :: NTuple{D,VF}, aabb_max :: NTuple{D,VF}, idx :: Int) where {D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
    # Contract:
    # - `idx` must directly index every AABB array.
    d2 = zero(TF)
    @inbounds for d in 1:D
        p = point[d]
        lo = aabb_min[d][idx]
        hi = aabb_max[d][idx]
        if p < lo
            Δ = lo - p
            d2 += Δ * Δ
        elseif p > hi
            Δ = p - hi
            d2 += Δ * Δ
        end
    end
    return d2
end

@inline function _squared_distance_point_line(point :: NTuple{D,TF}, origin :: NTuple{D,TF}, direction :: NTuple{D,TF}) where {D, TF <: AbstractFloat}
    # Contract:
    # - The line geometry is treated as an infinite line, not a ray.
    # - The returned value is the exact minimum squared Euclidean distance.
    # - `direction` must be a unit vector.
    zero_T = zero(TF)
    Δ2 = zero_T
    Δm = zero_T
    @inbounds for d in 1:D
        @inbounds begin
            p = point[d]
            o = origin[d]
            m = direction[d]
        end
        Δ = p - o
        Δ2 += Δ * Δ
        Δm += Δ * m
    end
    s = Δ2 - Δm * Δm
    return s
end

@inline function _line_intersects_aabb(origin :: NTuple{D,TF}, direction :: NTuple{D,TF}, aabb_min :: NTuple{D, VF} , aabb_max :: NTuple{D, VF}, idx :: Int) where {D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
    # Contract:
    # - The line geometry is treated as an infinite line, not a ray.
    # - `direction` must be a unit vector.
    # - `idx` must directly index every unified AABB array.
    tmin = typemin(TF)
    tmax = typemax(TF)

    @inbounds for d in 1:D
        @inbounds begin
            a = aabb_min[d][idx]
            b = aabb_max[d][idx]
            o = origin[d]
            m = direction[d]
        end

        if iszero(m)
            (o < a || o > b) && return false
        else
            t0 = (a - o) / m
            t1 = (b - o) / m

            if t0 > t1
                t0, t1 = t1, t0
            end

            tmin = max(tmin, t0)
            tmax = min(tmax, t1)

            (tmin > tmax) && return false
        end
    end

    return true
end

@inline function _squared_distance_line_aabb_lower_bound(origin :: NTuple{D,TF}, direction :: NTuple{D,TF}, aabb_min :: NTuple{D, VF} , aabb_max :: NTuple{D, VF}, idx :: Int) where {D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
    # Contract:
    # - The line geometry is treated as an infinite line, not a ray.
    # - `direction` must be a unit vector.
    # - The returned value is a conservative lower bound of the exact
    #   minimum squared Euclidean distance between the line and the AABB.
    # - `idx` must directly index every unified AABB array.

    zero_T = zero(TF)
    half_T = inv(TF(2))

    _line_intersects_aabb(origin, direction, aabb_min, aabb_max, idx) && return zero_T

    center = ntuple(d -> (aabb_min[d][idx] + aabb_max[d][idx]) * half_T, D)

    r2 = zero_T
    @inbounds for d in 1:D
        h = (aabb_max[d][idx] - aabb_min[d][idx]) * half_T
        r2 += h * h
    end

    dc2 = _squared_distance_point_line(center, origin, direction)

    if dc2 <= r2
        return zero_T
    else
        dc = sqrt(dc2)
        r  = sqrt(r2)
        δ  = dc - r
        return δ * δ
    end
end

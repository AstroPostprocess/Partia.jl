"""
    @LBVH_gather_point_traversal(LBVH, reference_point, radius2, leafsym, d2sym, leaf_hit)

Stackless DFS traversal over a `LinearBVH` using its `left` and `escape` links.
Traversal starts at unified node `1`; internal nodes occupy `1:nleaf-1` and
leaf nodes occupy `nleaf:2nleaf-1`.

This is the **gather** variant: the pruning radius is given by the `radius2` input.

# Arguments
- `LBVH`: Hierarchy with unified `aabb` storage and stackless topology.
- `reference_point`:
  `NTuple{D, TF}` query position.
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
directly by the input `radius2`. Internal nodes are pruned using the squared
line-to-AABB distance computed by `_squared_distance_line_aabb`. Leaf nodes
are also represented by AABBs and use their unified node ID for the same test.

# Parameters
- `LBVH`: Hierarchy with unified AABBs and stackless `left`/`escape` topology.
- `line_origin`:
  `NTuple{D, TF}` giving the origin of the query line.
- `line_direction`:
  `NTuple{D, TF}` giving the direction of the query line. This direction is
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
                $d2_ = _squared_distance_line_aabb($origin_, $direction_, $node_min_, $node_max_, Int($node_))
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
            $d2_node_ = _squared_distance_line_aabb($origin_, $direction_, $node_min_, $node_max_, $node_idx_)
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

Both leaf acceptance and internal pruning use the squared line-to-AABB
distance computed by `_squared_distance_line_aabb` over unified AABB storage.

# Parameters
- `LBVH`: Hierarchy holding unified `aabb`/`scale` arrays and `left`/`escape` topology.
- `line_origin`:
  `NTuple{D, TF}` giving the origin of the query line.
- `line_direction`:
  `NTuple{D, TF}` giving the direction of the query line. This direction is
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
                $d2_ = _squared_distance_line_aabb($origin_, $direction_, $node_min_, $node_max_, Int($node_))
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
            $d2_node_ = _squared_distance_line_aabb($origin_, $direction_, $node_min_, $node_max_, $node_idx_)
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

@inline function _squared_distance_point_aabb(point :: NTuple{D, TF}, aabb_min :: NTuple{D, VF}, aabb_max :: NTuple{D, VF}, idx :: Int) where {D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
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

@inline function _squared_distance_point_line(point :: NTuple{D, TF}, origin :: NTuple{D, TF}, direction :: NTuple{D, TF}) where {D, TF <: AbstractFloat}
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

@inline function _line_intersects_aabb(origin :: NTuple{D, TF}, direction :: NTuple{D, TF}, aabb_min :: NTuple{D, VF} , aabb_max :: NTuple{D, VF}, idx :: Int) where {D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
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

"""
Construct a symmetric box in line-origin-relative coordinates. Every subtraction
is enclosed outwards, including the distances from the rounded centre to the
endpoints. Thus, in real arithmetic, `c-e <= min-origin <= max-origin <= c+e`.
The returned box can be slightly larger than the input (also for point boxes).
"""
@inline function _line_aabb_local_box(origin :: NTuple{D, TF}, aabb_min :: NTuple{D, VF}, aabb_max :: NTuple{D, VF}, idx :: Int) where {D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
    bounds = ntuple(D) do d
        a = prevfloat(aabb_min[d][idx] - origin[d])
        b = nextfloat(aabb_max[d][idx] - origin[d])
        # A finite-input subtraction can exceed the floating-point range.
        # An unbounded axis still encloses that exact relative interval.
        if !isfinite(a) || !isfinite(b)
            return (zero(TF), TF(Inf))
        end
        # Halve first to avoid overflow in a+b. Extents below account for any
        # rounding here, including half-subnormal underflow.
        c = a / TF(2) + b / TF(2)
        e = max(nextfloat(c - a), nextfloat(b - c))
        (c, e)
    end
    return ntuple(d -> bounds[d][1], D), ntuple(d -> bounds[d][2], D)
end

# ============================================================
# 3D line–AABB distance (Eberly-style)
#
# Reference:
# David Eberly, Geometric Tools
# GTE/Mathematics/DistLine3CanonicalBox3.h
# https://github.com/davideberly/GeometricTools/blob/master/GTE/Mathematics/DistLine3CanonicalBox3.h
#
# Adapted for squared-distance-only queries.
# The edge calculations are simplified rather than
# a line-by-line translation of the original Face().
# ============================================================

@inline function _squared_distance_line_box_edge3d(p :: NTuple{3, TF}, v :: NTuple{3, TF}, e :: NTuple{3, TF}, i :: Int, j :: Int, k :: Int, bj :: TF, bk :: TF) where {TF <: AbstractFloat}
    # The edge varies along axis i.
    # Its fixed coordinates are q[j] = bj and q[k] = bk.
    # The direction v must be a unit vector.

    δj = p[j] - bj
    δk = p[k] - bk

    # Scale transverse direction before squaring; at least one component of w
    # is exactly one even when both input components are subnormal. No nonzero
    # component is classified as zero by an angular threshold.
    s = max(v[j], v[k])
    wj, wk = v[j] / s, v[k] / s
    den = muladd(wj, wj, wk * wk)
    h = max(abs(δj), abs(δk))
    h = iszero(h) ? one(TF) : h
    qj, qk = δj / h, δk / h
    tau = -muladd(wj, qj, wk * qk) / den # t*s/h, not t
    # Fast product tau*h*v[i]/s. Check every intermediate, since a
    # finite final result can hide earlier overflow or underflow to zero.
    vi = v[i]
    offset = if iszero(tau) || iszero(h) || iszero(vi)
        zero(TF)
    else
        th = tau * h
        thv = th * vi
        x = thv / s
        normal_min = floatmin(TF)
        safe = isfinite(th) & (abs(th) >= normal_min) &
               isfinite(thv) & (abs(thv) >= normal_min) &
               isfinite(x) & (abs(x) >= normal_min)
        if safe
            x
        else
            # Scale the ORIGINAL inputs; rounded intermediates may have
            # already lost information. floatmin is a format boundary,
            # not an angular/geometric epsilon. Do not use @fastmath here.
            mt, et = frexp(tau)
            mh, eh = frexp(h)
            mv, ev = frexp(vi)
            ms, es = frexp(s)
            ldexp(((mt * mh) * mv) / ms, et + eh + ev - es)
        end
    end
    u = p[i] + offset

    if -e[i] <= u <= e[i]
        # The closest point lies inside the edge
        dj = h * muladd(tau, wj, qj)
        dk = h * muladd(tau, wk, qk)
        return muladd(dj, dj, dk * dk)
    else
        # The closest point is an endpoint of the edge
        δi = p[i] - clamp(u, -e[i], e[i])
        h = max(abs(δi), abs(δj), abs(δk))
        iszero(h) && return zero(TF)
        qi, qj, qk = δi / h, δj / h, δk / h
        norm2 = muladd(v[i], v[i], muladd(v[j], v[j], v[k] * v[k]))
        tau = -muladd(v[i], qi, muladd(v[j], qj, v[k] * qk)) / norm2
        # A unit direction produced by floating-point normalization need not
        # have an exactly unit norm. Divide by its computed norm above.
        di = h * muladd(tau, v[i], qi)
        dj = h * muladd(tau, v[j], qj)
        dk = h * muladd(tau, v[k], qk)

        return muladd(di, di, muladd(dj, dj, dk * dk))
    end
end

@inline function _squared_distance_line_box_face3d(p :: NTuple{3, TF}, v :: NTuple{3, TF}, e :: NTuple{3, TF}, i0 :: Int, i1 :: Int, i2 :: Int) where {TF <: AbstractFloat}
    # The selected positive face is x[i0] = e[i0].
    # All components of v must be positive.

    pm = p[i0] - e[i0]

    inside1 = v[i0] * (p[i1] + e[i1]) >= v[i1] * pm
    inside2 = v[i0] * (p[i2] + e[i2]) >= v[i2] * pm

    if inside1 && inside2
        # The line intersects the box
        return zero(TF)
    elseif inside1
        # Closest point lies on the i1-edge or its endpoints
        return _squared_distance_line_box_edge3d(p, v, e, i1, i0, i2, e[i0], -e[i2])
    elseif inside2
        # Closest point lies on the i2-edge or its endpoints
        return _squared_distance_line_box_edge3d(p, v, e, i2, i0, i1, e[i0], -e[i1])
    else
        # Both negative-side edges are candidates
        d1 = _squared_distance_line_box_edge3d(p, v, e, i1, i0, i2, e[i0], -e[i2])
        d2 = _squared_distance_line_box_edge3d(p, v, e, i2, i0, i1, e[i0], -e[i1])
        return min(d1, d2)
    end
end

@inline function _squared_distance_line_box_2d(p :: NTuple{3, TF}, v :: NTuple{3, TF}, e :: NTuple{3, TF}, i :: Int, j :: Int, k :: Int) where {TF <: AbstractFloat}
    # v[i] > 0, v[j] > 0, v[k] == 0

    dk = max(abs(p[k]) - e[k], zero(TF))

    # Exact 2D line–rectangle distance
    sc = muladd(p[i], v[j], -p[j] * v[i])
    r = muladd(e[i], v[j], e[j] * v[i])

    δ = max(abs(sc) - r, zero(TF))
    return muladd(dk, dk, δ * δ / (v[i] * v[i] + v[j] * v[j]))
end

@inline function _squared_distance_line_box_1d(p :: NTuple{3, TF}, e :: NTuple{3, TF}, i :: Int, j :: Int) where {TF <: AbstractFloat}
    # Only the remaining axis has a nonzero direction

    di = max(abs(p[i]) - e[i], zero(TF))
    dj = max(abs(p[j]) - e[j], zero(TF))

    return muladd(di, di, dj * dj)
end

# 1D specialised implementation
@inline function _squared_distance_line_aabb(origin :: NTuple{1, TF}, direction :: NTuple{1, TF}, aabb_min :: NTuple{1, VF}, aabb_max :: NTuple{1, VF}, idx :: Int) where {TF <: AbstractFloat, VF <: AbstractVector{TF}}
    # A nonzero direction spans the entire real line.
    !iszero(direction[1]) && return zero(TF)

    # Degenerate zero direction; outside the unit-vector contract.
    # Use the same outward-enclosed local box as the 2D/3D methods.
    c, e = _line_aabb_local_box(origin, aabb_min, aabb_max, idx)
    δ = max(abs(c[1]) - e[1], zero(TF))
    return δ * δ
end

# 2D specialised implementation
@inline function _squared_distance_line_aabb(origin :: NTuple{2, TF}, direction :: NTuple{2, TF}, aabb_min :: NTuple{2, VF}, aabb_max :: NTuple{2, VF}, idx :: Int) where {TF <: AbstractFloat, VF <: AbstractVector{TF}}
    zero_T = zero(TF)

    c, e = _line_aabb_local_box(origin, aabb_min, aabb_max, idx)

    # Unit normal to the line
    nx, ny = direction[2], -direction[1]

    # Signed centre projection and projected half extent
    sc = muladd(c[1], nx, c[2] * ny)
    # An unbounded axis parallel to the line contributes zero, not Inf*0.
    rx = iszero(nx) ? zero_T : e[1] * abs(nx)
    ry = iszero(ny) ? zero_T : e[2] * abs(ny)
    r = rx + ry

    δ = max(abs(sc) - r, zero_T)
    return δ * δ
end

# 3D specialised implementation
@inline function _squared_distance_line_aabb(origin :: NTuple{3, TF}, direction :: NTuple{3, TF}, aabb_min :: NTuple{3, VF}, aabb_max :: NTuple{3, VF}, idx :: Int) where {TF <: AbstractFloat, VF <: AbstractVector{TF}}
    zero_T = zero(TF)

    c, e = _line_aabb_local_box(origin, aabb_min, aabb_max, idx)
    o = ntuple(d -> -c[d], 3)

    # Reflect direction to the positive octant
    p = ntuple(d -> direction[d] < zero_T ? -o[d] : o[d], 3)
    v = ntuple(d -> abs(direction[d]), 3)

    if v[1] > zero_T
        if v[2] > zero_T
            if v[3] > zero_T
                # Select the first positive face, following Eberly
                pm = ntuple(d -> p[d] - e[d], 3)

                if v[2] * pm[1] >= v[1] * pm[2]
                    if v[3] * pm[1] >= v[1] * pm[3]
                        return _squared_distance_line_box_face3d(p, v, e, 1, 2, 3)
                    else
                        return _squared_distance_line_box_face3d(p, v, e, 3, 1, 2)
                    end
                else
                    if v[3] * pm[2] >= v[2] * pm[3]
                        return _squared_distance_line_box_face3d(p, v, e, 2, 3, 1)
                    else
                        return _squared_distance_line_box_face3d(p, v, e, 3, 1, 2)
                    end
                end
            else
                return _squared_distance_line_box_2d(p, v, e, 1, 2, 3)
            end
        elseif v[3] > zero_T
            return _squared_distance_line_box_2d(p, v, e, 1, 3, 2)
        else
            return _squared_distance_line_box_1d(p, e, 2, 3)
        end
    elseif v[2] > zero_T
        if v[3] > zero_T
            return _squared_distance_line_box_2d(p, v, e, 2, 3, 1)
        else
            return _squared_distance_line_box_1d(p, e, 1, 3)
        end
    elseif v[3] > zero_T
        return _squared_distance_line_box_1d(p, e, 1, 2)
    else
        # Degenerate zero direction; outside the unit-vector contract
        d2 = zero_T
        for d in 1:3
            δ = max(abs(p[d]) - e[d], zero_T)
            d2 = muladd(δ, δ, d2)
        end
        return d2
    end
end

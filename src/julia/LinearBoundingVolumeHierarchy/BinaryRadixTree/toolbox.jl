######################################################################################

# Binary radix tree helper routines for unified node IDs, Karras range/split
# construction, and stackless escape-link generation.
#     by Wei-Shan Su,
#     May 4, 2026

######################################################################################
@inline is_leaf_id(node :: Int32, nleaf :: Int) = (node != 0) & (node >= Int32(nleaf))
@inline is_internal_id(node :: Int32, nleaf :: Int) = (node != 0) & (node < Int32(nleaf))
@inline leaf_index(node :: Int32, nleaf :: Int) = Int(node) - (nleaf - 1)               # 1..nleaf
@inline internal_index(node :: Int32) = Int(node)                                     # 1..ninternal

@inline function _range_direction(codes :: V, i :: Int) where {TI <: Unsigned, V <: AbstractVector{TI}}
    δL = (i > 1) ?  _longest_common_prefix_length(codes, i, i - 1) : -1
    δR = (i < length(codes)) ? _longest_common_prefix_length(codes, i, i + 1) : -1
    d = sign(δR - δL)
    return (d == 0) ? 1 : d
end


@inline function _find_range(codes :: V, i :: Int) where {TI <: Unsigned, V <: AbstractVector{TI}}
    n = length(codes)
    d = _range_direction(codes, i)
    δmin = (1 <= i - d <= length(codes)) ? _longest_common_prefix_length(codes, i, i - d) : -1

    # Find the upper limit
    lmax = 2
    while true
        j = i + lmax * d
        if (j < 1) || (j > n)
            break
        end
        δtest = _longest_common_prefix_length(codes, i, j)
        if δtest <= δmin
            break
        end
        lmax *= 2
    end
    # Find the other side
    l = 0
    t = lmax >> 1    # lmax / 2, still Int
    while t > 0
        j = i + (l + t) * d
        if j >= 1 && j <= n && _longest_common_prefix_length(codes, i, j) > δmin
            l += t
        end
        t >>= 1      # t = t / 2, still Int
    end

    j = i + l * d
    return (min(i, j), max(i, j))
end

@inline function _split_position(codes :: V, first :: Int, last :: Int) where {TI <: Unsigned, V <: AbstractVector{TI}}
    if codes[first] == codes[last]
        return clamp((first + last) >> 1, first, last - 1)
    end

    @inbounds prefix_first_last = _longest_common_prefix_length(codes, first, last)
    split = first
    step = last - first

    while step > 1
        step = (step + 1) >> 1
        new_split = split + step
        if new_split < last
            prefix = _longest_common_prefix_length(codes, first, new_split)
            if prefix > prefix_first_last
                split = new_split
            end
        end
    end

    return split
end

@inline function _build_adjacent!(adj :: S, codes :: V, i :: Int) where {TI <: Unsigned, V <: AbstractVector{TI},  S <: AbstractVector{Int32}}
    adj[i] = _longest_common_prefix_length(codes, i)
    return nothing
end

@inline function _build_child!(left :: S, right :: S, range_hi :: S, parent :: S, codes :: V, i :: Int) where {TI <: Unsigned, V <: AbstractVector{TI},  S <: AbstractVector{Int32}}
    n = length(codes)
    leaf_offset = n - 1

    @inbounds begin
        lo, hi = _find_range(codes, i)
        range_hi[i] = Int32(hi)
        s = _split_position(codes, lo, hi)

        # NOTE (threaded build): We assume each child node has a unique parent in a valid BRT.
        # If enc.codes is not sorted, or duplicate-code tie-breaking is broken, the topology may
        # become invalid and the same `idx` could be assigned by multiple `i` concurrently,
        # causing nondeterministic overwrites in `parent[idx]`.

        # left child
        if s == lo
            idx = lo + leaf_offset              # leaf node id in n..2n-1
        else
            idx = s                             # internal node id in 1..n-1
        end
        left[i] = Int32(idx)
        parent[idx] = Int32(i)

        # right child
        if s + 1 == hi
            idx = hi + leaf_offset
        else
            idx = s + 1
        end
        right[i] = Int32(idx)
        parent[idx] = Int32(i)
    end

    return nothing
end

@inline function _build_escape!(escape :: V, adj :: V, range_hi :: V, nleaf :: Int, i :: Int) where {V <: AbstractVector{Int32}}
    # Escape for internal node
    leaf_offset = nleaf - 1
    @inbounds begin
        hi = Int(range_hi[i])   # hi is leaf index in 1..nleaf
        if hi == nleaf
            escape[i] = 0
        elseif hi == nleaf - 1
            escape[i] = Int32(leaf_offset + nleaf)
        else
            next = hi + 1
            escape[i] = (adj[next] < adj[next-1]) ? Int32(leaf_offset + next) : Int32(next)
        end
    end
    return nothing
end

@inline function _build_escape!(escape :: V, parent :: V, left :: V, right :: V, i :: Int) where {V <: AbstractVector{Int32}}
    # Escape for leafs
    @inbounds begin
        p = parent[i]              # unified internal id (1..nleaf-1) or 0
        if iszero(p)
            escape[i] = 0
            return nothing
        end

        pidx = internal_index(p)         # 1..ninternal
        if left[pidx] == Int32(i)
            # next is the sibling subtree root
            escape[i] = right[pidx]
        else
            # must be right child => go where parent would escape to
            # p is unified id, so escape[p] is valid
            escape[i] = escape[Int(p)]
        end
    end
    return nothing
end



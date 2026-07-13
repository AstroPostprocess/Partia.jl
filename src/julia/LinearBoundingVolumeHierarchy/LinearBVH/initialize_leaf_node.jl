@inline function _initialize_leaf_node!(unified_scale :: Vector{TF}, aabb :: AABB{D, TF, Vector{TF}}, scale :: Vector{TF}, leaf_min :: NTuple{D, Vector{TF}}, leaf_max :: NTuple{D, Vector{TF}}, leaf_offset :: Int, i :: Int) where {D, TF <: AbstractFloat}
    leaf_idx = i + leaf_offset

    @inbounds begin
        unified_scale[leaf_idx] = scale[i]
        for d in 1:D
            lmin = leaf_min[d][i]
            lmax = leaf_max[d][i]
            lmin <= lmax || throw(ArgumentError("LinearBVH: leaf_min[$d][$i] = $lmin exceeds leaf_max[$d][$i] = $lmax."))
            aabb.min[d][leaf_idx] = lmin
            aabb.max[d][leaf_idx] = lmax
        end
    end

    return nothing
end

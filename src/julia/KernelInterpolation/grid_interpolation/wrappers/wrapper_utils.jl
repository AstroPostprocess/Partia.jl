######################################################################################

#     Grid interpolation wrapper utilities

######################################################################################
@inline _validate_interpolation_lbvh_leaf_order(input, LBVH) = nothing

@inline function _validate_interpolation_lbvh_leaf_order(input :: AbstractInterpolationInput{D, TF, Vector{TF}}, LBVH :: LinearBVH{D, TF, Vector{TF}}) where {D, TF <: AbstractFloat}
    matches_lbvh_leaf_order(input, LBVH) || throw(ArgumentError(
        "Provided LBVH leaf order does not match the current input ordering. " *
        "Ensure the LBVH was built from the same Morton-reordered input."
    ))
    return nothing
end

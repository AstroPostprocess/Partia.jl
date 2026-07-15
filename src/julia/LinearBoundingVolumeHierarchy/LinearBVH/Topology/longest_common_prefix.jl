######################################################################################

# Longest-common-prefix helpers used by the bottom-up LBVH constructor.
#     by Wei-Shan Su,
#     July 13, 2026

######################################################################################
@inline function _longest_common_prefix_length(a :: T, b :: T) :: Int where {T <: Unsigned}
    return leading_zeros(xor(a, b))
end

@inline function _longest_common_prefix_length(codes :: V, i :: Int, j :: Int) :: Int where {T <: Unsigned, V <: AbstractVector{T}}
    @inbounds a, b = codes[i], codes[j]
    return a == b ? 8 * sizeof(T) + _longest_common_prefix_length(UInt64(i), UInt64(j)) :
                    _longest_common_prefix_length(a, b)
end

@inline _longest_common_prefix_length(codes :: V, i :: Int) where {T <: Unsigned, V <: AbstractVector{T}} =
    _longest_common_prefix_length(codes, i, i + 1)

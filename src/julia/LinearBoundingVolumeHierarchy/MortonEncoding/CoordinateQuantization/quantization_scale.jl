######################################################################################

# Coordinate quantization scale helpers.
#     by Wei-Shan Su,
#     July 12, 2026

######################################################################################
"""
    _axis_bits(::Val{D}, ::Type{T})

Return the number of coordinate bits available per axis for `D` dimensions.
"""
@inline function _axis_bits(::Val{D}, ::Type{T}) where {D, T <: Unsigned}
    nbits = sizeof(T) * 8
    return div(nbits - (nbits % D), D)
end

"""
    _axis_scale(::Val{D}, ::Type{T}, ::Type{TF})

Return the scale used to map normalized coordinates onto integer bins.
"""
@inline function _axis_scale(::Val{D}, ::Type{T}, ::Type{TF}) where {D, T <: Unsigned, TF <: AbstractFloat}
    bits = _axis_bits(Val(D), T)
    return exp2(TF(bits)) - one(TF)
end

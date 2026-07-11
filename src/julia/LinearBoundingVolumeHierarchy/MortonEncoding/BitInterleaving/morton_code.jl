######################################################################################

# Morton code encoding.
#     by Wei-Shan Su,
#     July 12, 2026

######################################################################################
"""
    _encode_morton_code2D(ix, iy)

Interleave one quantized 2D integer coordinate into a Morton code.
"""
@inline function _encode_morton_code2D(ix :: T, iy :: T) where {T <: Unsigned}
    return (_expand_bits2D(ix) << 1) | _expand_bits2D(iy)
end

"""
    _encode_morton_code3D(ix, iy, iz)

Interleave one quantized 3D integer coordinate into a Morton code.
"""
@inline function _encode_morton_code3D(ix :: T, iy :: T, iz :: T) where {T <: Unsigned}
    ex = _expand_bits3D(ix)
    ey = _expand_bits3D(iy)
    ez = _expand_bits3D(iz)
    return (ex << 2) | (ey << 1) | ez
end

######################################################################################

# Bit spreading primitives for Morton interleaving.
#     by Wei-Shan Su,
#     July 12, 2026

# The 3D expansion masks are adapted from:
# https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints

######################################################################################
"""
    _expand_bits3D(x :: UInt32)

Spread the low bits of a 32-bit integer for 3D Morton interleaving.
"""
@inline function _expand_bits3D(x :: UInt32)
    x = (x | (x << 16)) & 0x30000ff
    x = (x | (x << 8)) & 0x300f00f
    x = (x | (x << 4)) & 0x30c30c3
    x = (x | (x << 2)) & 0x9249249
    return x
end

"""
    _expand_bits3D(x :: UInt64)

Spread the low bits of a 64-bit integer for 3D Morton interleaving.
"""
@inline function _expand_bits3D(x :: UInt64)
    x = (x | (x << 32)) & 0x1f00000000ffff
    x = (x | (x << 16)) & 0x1f0000ff0000ff
    x = (x | (x << 8)) & 0x100f00f00f00f00f
    x = (x | (x << 4)) & 0x10c30c30c30c30c3
    x = (x | (x << 2)) & 0x1249249249249249
    return x
end

"""
    _expand_bits2D(x :: UInt32)

Spread the low bits of a 32-bit integer for 2D Morton interleaving.
"""
@inline function _expand_bits2D(x :: UInt32)
    x = (x | (x << 8)) & 0x00ff00ff
    x = (x | (x << 4)) & 0x0f0f0f0f
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    return x
end

"""
    _expand_bits2D(x :: UInt64)

Spread the low bits of a 64-bit integer for 2D Morton interleaving.
"""
@inline function _expand_bits2D(x :: UInt64)
    x = (x | (x << 16)) & 0x0000ffff0000ffff
    x = (x | (x << 8)) & 0x00ff00ff00ff00ff
    x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0f
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555
    return x
end

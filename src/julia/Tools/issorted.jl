function _issorted(array :: Vector{T}) :: Bool where {T <: Unsigned}
    return issorted(array)
end
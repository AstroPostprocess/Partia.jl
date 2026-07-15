function Partia.LineSamples(xo :: CuVector{T}, yo :: CuVector{T}, zo :: CuVector{T}, xd :: CuVector{T}, yd :: CuVector{T}, zd :: CuVector{T}) where {T <: AbstractFloat}
    origin = (xo, yo, zo)
    direction = (xd, yd, zd)
    vals = CUDA.zeros(T, length(xo))
    return LineSamples(vals, origin, direction)
end

function Partia.LineSamples(xo :: CuVector{T}, yo :: CuVector{T}, xd :: CuVector{T}, yd :: CuVector{T}) where {T <: AbstractFloat}
    origin = (xo, yo)
    direction = (xd, yd)
    vals = CUDA.zeros(T, length(xo))
    return LineSamples(vals, origin, direction)
end

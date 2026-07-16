function Partia.PointSamples(x :: CuVector{T}, y :: CuVector{T}, z :: CuVector{T}) where {T <: AbstractFloat}
    coords = (x, y, z)
    vals = CUDA.zeros(T, length(x))
    return PointSamples(vals, coords)
end

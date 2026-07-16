function Partia.PointSamples(x :: MtlVector{Float32}, y :: MtlVector{Float32}, z :: MtlVector{Float32})
    coords = (x, y, z)
    vals = Metal.zeros(Float32, length(x))
    return PointSamples(vals, coords)
end

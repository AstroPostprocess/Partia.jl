function Partia.LineSamples(xo :: MtlVector{Float32}, yo :: MtlVector{Float32}, zo :: MtlVector{Float32}, xd :: MtlVector{Float32}, yd :: MtlVector{Float32}, zd :: MtlVector{Float32})
    origin = (xo, yo, zo)
    direction = (xd, yd, zd)
    vals = Metal.zeros(Float32, length(xo))
    return LineSamples(vals, origin, direction)
end

function Partia.LineSamples(xo :: MtlVector{Float32}, yo :: MtlVector{Float32}, xd :: MtlVector{Float32}, yd :: MtlVector{Float32})
    origin = (xo, yo)
    direction = (xd, yd)
    vals = Metal.zeros(Float32, length(xo))
    return LineSamples(vals, origin, direction)
end

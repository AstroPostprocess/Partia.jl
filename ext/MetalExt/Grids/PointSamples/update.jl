"""
    update!(sample, coordinate_system, frame, axis_parameters..., ::Val{ThreadsPerGroup}=Val(256))

Resize Metal point-sample storage and rebuild it with the corresponding `build!`
method. Geometry is regenerated from the host frame and values are cleared.

# Parameters
- `sample`: Reusable Metal point-sample storage.
- `coordinate_system`: Coordinate-system dispatch tag.
- `frame`: Host frame supplying position and basis vectors.
- `axis_parameters`: Constructor-compatible axis specifications.
- `::Val{ThreadsPerGroup}`: Metal threads per threadgroup.

# Returns
- `PointSamples`: The same `sample` wrapper.
"""
function Partia.update!(sample :: PointSamples{3, Float32, MtlVector{Float32}}, coordinate_system :: Type{Cartesian}, frame :: Frame{Float32}, xparams :: AxisParam{Float32}, yparams :: AxisParam{Float32}, threads :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup}
    _resize_point_samples!(sample, xparams[3] * yparams[3])
    Partia.build!(sample, coordinate_system, frame, xparams, yparams, threads)
    return sample
end

function Partia.update!(sample :: PointSamples{3, Float32, MtlVector{Float32}}, coordinate_system :: Type{Cartesian}, frame :: Frame{Float32}, xparams :: AxisParam{Float32}, yparams :: AxisParam{Float32}, zparams :: AxisParam{Float32}, threads :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup}
    _resize_point_samples!(sample, xparams[3] * yparams[3] * zparams[3])
    Partia.build!(sample, coordinate_system, frame, xparams, yparams, zparams, threads)
    return sample
end

function Partia.update!(sample :: PointSamples{3, Float32, MtlVector{Float32}}, coordinate_system :: Type{Polar}, frame :: Frame{Float32}, sparams :: AxisParam{Float32}, ϕparams :: AxisParam{Float32}, threads :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup}
    _resize_point_samples!(sample, sparams[3] * ϕparams[3])
    Partia.build!(sample, coordinate_system, frame, sparams, ϕparams, threads)
    return sample
end

function Partia.update!(sample :: PointSamples{3, Float32, MtlVector{Float32}}, coordinate_system :: Type{Cylindrical}, frame :: Frame{Float32}, sparams :: AxisParam{Float32}, ϕparams :: AxisParam{Float32}, zparams :: AxisParam{Float32}, threads :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup}
    _resize_point_samples!(sample, sparams[3] * ϕparams[3] * zparams[3])
    Partia.build!(sample, coordinate_system, frame, sparams, ϕparams, zparams, threads)
    return sample
end

@inline function _resize_point_samples!(sample :: PointSamples{3, Float32, MtlVector{Float32}}, n :: Int) 
    resize!(sample.grid, n)
    @inbounds for d in 1:3
        resize!(sample.coor[d], n)
    end
    return nothing
end

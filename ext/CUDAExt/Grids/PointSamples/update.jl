"""
    update!(sample, coordinate_system, frame, axis_parameters..., ::Val{ThreadsPerBlock}=Val(256))

Resize CUDA point-sample storage and rebuild it with the corresponding `build!`
method. Geometry is regenerated from the host frame and values are cleared.

# Parameters
- `sample`: Reusable CUDA point-sample storage.
- `coordinate_system`: Coordinate-system dispatch tag.
- `frame`: Host frame supplying position and basis vectors.
- `axis_parameters`: Constructor-compatible axis specifications.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `PointSamples`: The same `sample` wrapper.
"""
function Partia.update!(sample :: PointSamples{3, TF, CuVector{TF}}, coordinate_system :: Type{Cartesian}, frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}, threads :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, ThreadsPerBlock}
    _resize_point_samples!(sample, xparams[3] * yparams[3])
    Partia.build!(sample, coordinate_system, frame, xparams, yparams, threads)
    return sample
end

function Partia.update!(sample :: PointSamples{3, TF, CuVector{TF}}, coordinate_system :: Type{Cartesian}, frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}, zparams :: AxisParam{TF}, threads :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, ThreadsPerBlock}
    _resize_point_samples!(sample, xparams[3] * yparams[3] * zparams[3])
    Partia.build!(sample, coordinate_system, frame, xparams, yparams, zparams, threads)
    return sample
end

function Partia.update!(sample :: PointSamples{3, TF, CuVector{TF}}, coordinate_system :: Type{Polar}, frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}, threads :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, ThreadsPerBlock}
    _resize_point_samples!(sample, sparams[3] * ϕparams[3])
    Partia.build!(sample, coordinate_system, frame, sparams, ϕparams, threads)
    return sample
end

function Partia.update!(sample :: PointSamples{3, TF, CuVector{TF}}, coordinate_system :: Type{Cylindrical}, frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}, zparams :: AxisParam{TF}, threads :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, ThreadsPerBlock}
    _resize_point_samples!(sample, sparams[3] * ϕparams[3] * zparams[3])
    Partia.build!(sample, coordinate_system, frame, sparams, ϕparams, zparams, threads)
    return sample
end

@inline function _resize_point_samples!(sample :: PointSamples{3, TF, CuVector{TF}}, n :: Int) where {TF <: AbstractFloat}
    resize!(sample.grid, n)
    @inbounds for d in 1:3
        resize!(sample.coor[d], n)
    end
    return nothing
end

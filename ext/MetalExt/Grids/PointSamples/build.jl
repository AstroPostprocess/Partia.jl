"""
    build!(sample, Cartesian, frame, xparams, yparams[, zparams], ::Val{ThreadsPerGroup}=Val(256))
    build!(sample, Polar, frame, sparams, ϕparams, ::Val{ThreadsPerGroup}=Val(256))
    build!(sample, Cylindrical, frame, sparams, ϕparams, zparams, ::Val{ThreadsPerGroup}=Val(256))

Rebuild preallocated Metal point-sample geometry from a host `Frame`. Frame
position and basis vectors are passed to a one-dimensional device kernel as
scalar tuples. Storage size must already match, and `sample.grid` is cleared.

# Parameters
- `sample`: Preallocated Metal point-sample storage.
- `Cartesian`, `Polar`, `Cylindrical`: Coordinate-system dispatch tag.
- `frame`: Host frame supplying position and basis vectors.
- `xparams`, `yparams`, `zparams`: Cartesian axis specifications.
- `sparams`, `ϕparams`: Radial and angular axis specifications.
- `::Val{ThreadsPerGroup}`: Metal threads per threadgroup.

# Returns
- `nothing`: Device geometry and values are updated in place.
"""
function Partia.build!(sample :: PointSamples{3, Float32, MtlVector{Float32}}, :: Type{Cartesian}, frame :: Frame{Float32}, xparams :: AxisParam{Float32}, yparams :: AxisParam{Float32}, :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup}
    n = xparams[3] * yparams[3]
    length(sample) == n || throw(DimensionMismatch("sample storage must have length $n"))
    @metal threads=(ThreadsPerGroup,) groups=(cld(n, ThreadsPerGroup),) _cartesian_plane_point_samples_kernel!(sample.grid, sample.coor, frame_position(frame), frame_right(frame), frame_up(frame), xparams, yparams)
    return nothing
end

function Partia.build!(sample :: PointSamples{3, Float32, MtlVector{Float32}}, :: Type{Cartesian}, frame :: Frame{Float32}, xparams :: AxisParam{Float32}, yparams :: AxisParam{Float32}, zparams :: AxisParam{Float32}, :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup}
    n = xparams[3] * yparams[3] * zparams[3]
    length(sample) == n || throw(DimensionMismatch("sample storage must have length $n"))
    @metal threads=(ThreadsPerGroup,) groups=(cld(n, ThreadsPerGroup),) _cartesian_box_point_samples_kernel!(sample.grid, sample.coor, frame_position(frame), frame_right(frame), frame_up(frame), frame_forward(frame), xparams, yparams, zparams)
    return nothing
end

function Partia.build!(sample :: PointSamples{3, Float32, MtlVector{Float32}}, :: Type{Polar}, frame :: Frame{Float32}, sparams :: AxisParam{Float32}, ϕparams :: AxisParam{Float32}, :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup}
    n = sparams[3] * ϕparams[3]
    length(sample) == n || throw(DimensionMismatch("sample storage must have length $n"))
    @metal threads=(ThreadsPerGroup,) groups=(cld(n, ThreadsPerGroup),) _polar_point_samples_kernel!(sample.grid, sample.coor, frame_position(frame), frame_right(frame), frame_up(frame), sparams, ϕparams)
    return nothing
end

function Partia.build!(sample :: PointSamples{3, Float32, MtlVector{Float32}}, :: Type{Cylindrical}, frame :: Frame{Float32}, sparams :: AxisParam{Float32}, ϕparams :: AxisParam{Float32}, zparams :: AxisParam{Float32}, :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup}
    n = sparams[3] * ϕparams[3] * zparams[3]
    length(sample) == n || throw(DimensionMismatch("sample storage must have length $n"))
    @metal threads=(ThreadsPerGroup,) groups=(cld(n, ThreadsPerGroup),) _cylindrical_point_samples_kernel!(sample.grid, sample.coor, frame_position(frame), frame_right(frame), frame_up(frame), frame_forward(frame), sparams, ϕparams, zparams)
    return nothing
end

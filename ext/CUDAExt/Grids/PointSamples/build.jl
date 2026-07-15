"""
    build!(sample, Cartesian, frame, xparams, yparams[, zparams], ::Val{ThreadsPerBlock}=Val(256))
    build!(sample, Polar, frame, sparams, ϕparams, ::Val{ThreadsPerBlock}=Val(256))
    build!(sample, Cylindrical, frame, sparams, ϕparams, zparams, ::Val{ThreadsPerBlock}=Val(256))

Rebuild preallocated CUDA point-sample geometry from a host `Frame`. Frame
position and basis vectors are passed to a one-dimensional device kernel as
scalar tuples. Storage size must already match, and `sample.grid` is cleared.

# Parameters
- `sample`: Preallocated CUDA point-sample storage.
- `Cartesian`, `Polar`, `Cylindrical`: Coordinate-system dispatch tag.
- `frame`: Host frame supplying position and basis vectors.
- `xparams`, `yparams`, `zparams`: Cartesian axis specifications.
- `sparams`, `ϕparams`: Radial and angular axis specifications.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `nothing`: Device geometry and values are updated in place.
"""
function Partia.build!(sample :: PointSamples{3, TF, CuVector{TF}}, :: Type{Cartesian}, frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}, :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, ThreadsPerBlock}
    n = xparams[3] * yparams[3]
    length(sample) == n || throw(DimensionMismatch("sample storage must have length $n"))
    @cuda threads=ThreadsPerBlock blocks=cld(n, ThreadsPerBlock) _cartesian_plane_point_samples_kernel!(sample.grid, sample.coor, frame_position(frame), frame_right(frame), frame_up(frame), xparams, yparams)
    return nothing
end

function Partia.build!(sample :: PointSamples{3, TF, CuVector{TF}}, :: Type{Cartesian}, frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}, zparams :: AxisParam{TF}, :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, ThreadsPerBlock}
    n = xparams[3] * yparams[3] * zparams[3]
    length(sample) == n || throw(DimensionMismatch("sample storage must have length $n"))
    @cuda threads=ThreadsPerBlock blocks=cld(n, ThreadsPerBlock) _cartesian_box_point_samples_kernel!(sample.grid, sample.coor, frame_position(frame), frame_right(frame), frame_up(frame), frame_forward(frame), xparams, yparams, zparams)
    return nothing
end

function Partia.build!(sample :: PointSamples{3, TF, CuVector{TF}}, :: Type{Polar}, frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}, :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, ThreadsPerBlock}
    n = sparams[3] * ϕparams[3]
    length(sample) == n || throw(DimensionMismatch("sample storage must have length $n"))
    @cuda threads=ThreadsPerBlock blocks=cld(n, ThreadsPerBlock) _polar_point_samples_kernel!(sample.grid, sample.coor, frame_position(frame), frame_right(frame), frame_up(frame), sparams, ϕparams)
    return nothing
end

function Partia.build!(sample :: PointSamples{3, TF, CuVector{TF}}, :: Type{Cylindrical}, frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}, zparams :: AxisParam{TF}, :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, ThreadsPerBlock}
    n = sparams[3] * ϕparams[3] * zparams[3]
    length(sample) == n || throw(DimensionMismatch("sample storage must have length $n"))
    @cuda threads=ThreadsPerBlock blocks=cld(n, ThreadsPerBlock) _cylindrical_point_samples_kernel!(sample.grid, sample.coor, frame_position(frame), frame_right(frame), frame_up(frame), frame_forward(frame), sparams, ϕparams, zparams)
    return nothing
end

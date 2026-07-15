"""
    build!(sample, Cartesian, ParallelBeam, frame, xparams, yparams, ::Val{ThreadsPerGroup}=Val(256))
    build!(sample, Polar, ParallelBeam, frame, sparams, ϕparams, ::Val{ThreadsPerGroup}=Val(256))

Rebuild preallocated Metal parallel-beam line samples from a host `Frame`.
Origins, directions, and zero values are written by a one-dimensional kernel.

# Parameters
- `sample`: Preallocated Metal line-sample storage.
- `Cartesian`, `Polar`: Coordinate-system dispatch tag.
- `ParallelBeam`: Parallel-beam model dispatch tag.
- `frame`: Host frame supplying position and basis vectors.
- `xparams`, `yparams`, `sparams`, `ϕparams`: Plane axis specifications.
- `::Val{ThreadsPerGroup}`: Metal threads per threadgroup.

# Returns
- `nothing`: Device geometry and values are updated in place.
"""
function Partia.build!(sample :: LineSamples{3, Float32, MtlVector{Float32}}, coordinate_system :: Type{Cartesian}, :: Type{ParallelBeam}, frame :: Frame{Float32}, xparams :: AxisParam{Float32}, yparams :: AxisParam{Float32}, :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup}
    n = xparams[3] * yparams[3]
    length(sample) == n || throw(DimensionMismatch("sample storage must have length $n"))
    @metal threads=(ThreadsPerGroup,) groups=(cld(n, ThreadsPerGroup),) _parallel_beam_line_samples_kernel!(sample.grid, sample.origin, sample.direction, frame_position(frame), frame_right(frame), frame_up(frame), frame_forward(frame), xparams, yparams, coordinate_system)
    return nothing
end

function Partia.build!(sample :: LineSamples{3, Float32, MtlVector{Float32}}, coordinate_system :: Type{Polar}, :: Type{ParallelBeam}, frame :: Frame{Float32}, sparams :: AxisParam{Float32}, ϕparams :: AxisParam{Float32}, :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup}
    n = sparams[3] * ϕparams[3]
    length(sample) == n || throw(DimensionMismatch("sample storage must have length $n"))
    @metal threads=(ThreadsPerGroup,) groups=(cld(n, ThreadsPerGroup),) _parallel_beam_line_samples_kernel!(sample.grid, sample.origin, sample.direction, frame_position(frame), frame_right(frame), frame_up(frame), frame_forward(frame), sparams, ϕparams, coordinate_system)
    return nothing
end

"""
    build!(sample, Cartesian, ParallelBeam, frame, xparams, yparams, ::Val{ThreadsPerBlock}=Val(256))
    build!(sample, Polar, ParallelBeam, frame, sparams, ϕparams, ::Val{ThreadsPerBlock}=Val(256))

Rebuild preallocated CUDA parallel-beam line samples from a host `Frame`.
Origins, directions, and zero values are written by a one-dimensional kernel.

# Parameters
- `sample`: Preallocated CUDA line-sample storage.
- `Cartesian`, `Polar`: Coordinate-system dispatch tag.
- `ParallelBeam`: Parallel-beam model dispatch tag.
- `frame`: Host frame supplying position and basis vectors.
- `xparams`, `yparams`, `sparams`, `ϕparams`: Plane axis specifications.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `nothing`: Device geometry and values are updated in place.
"""
function Partia.build!(sample :: LineSamples{3, TF, CuVector{TF}}, coordinate_system :: Type{Cartesian}, :: Type{ParallelBeam}, frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}, :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, ThreadsPerBlock}
    n = xparams[3] * yparams[3]
    length(sample) == n || throw(DimensionMismatch("sample storage must have length $n"))
    @cuda threads=ThreadsPerBlock blocks=cld(n, ThreadsPerBlock) _parallel_beam_line_samples_kernel!(sample.grid, sample.origin, sample.direction, frame_position(frame), frame_right(frame), frame_up(frame), frame_forward(frame), xparams, yparams, coordinate_system)
    return nothing
end

function Partia.build!(sample :: LineSamples{3, TF, CuVector{TF}}, coordinate_system :: Type{Polar}, :: Type{ParallelBeam}, frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}, :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, ThreadsPerBlock}
    n = sparams[3] * ϕparams[3]
    length(sample) == n || throw(DimensionMismatch("sample storage must have length $n"))
    @cuda threads=ThreadsPerBlock blocks=cld(n, ThreadsPerBlock) _parallel_beam_line_samples_kernel!(sample.grid, sample.origin, sample.direction, frame_position(frame), frame_right(frame), frame_up(frame), frame_forward(frame), sparams, ϕparams, coordinate_system)
    return nothing
end

"""
    update!(sample, coordinate_system, ParallelBeam, frame, axis1, axis2, ::Val{ThreadsPerBlock}=Val(256))

Resize CUDA line-sample storage and rebuild parallel-beam geometry. The same
wrapper is returned and all sample values are cleared by `build!`.

# Parameters
- `sample`: Reusable CUDA line-sample storage.
- `coordinate_system`: Cartesian or polar dispatch tag.
- `ParallelBeam`: Parallel-beam model dispatch tag.
- `frame`: Host frame supplying position and basis vectors.
- `axis1`, `axis2`: Constructor-compatible plane axis specifications.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `LineSamples`: The same `sample` wrapper.
"""
function Partia.update!(sample :: LineSamples{3, TF, CuVector{TF}}, coordinate_system :: Type{Cartesian}, beam_model :: Type{ParallelBeam}, frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}, threads :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, ThreadsPerBlock}
    _resize_line_samples!(sample, xparams[3] * yparams[3])
    Partia.build!(sample, coordinate_system, beam_model, frame, xparams, yparams, threads)
    return sample
end

function Partia.update!(sample :: LineSamples{3, TF, CuVector{TF}}, coordinate_system :: Type{Polar}, beam_model :: Type{ParallelBeam}, frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}, threads :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, ThreadsPerBlock}
    _resize_line_samples!(sample, sparams[3] * ϕparams[3])
    Partia.build!(sample, coordinate_system, beam_model, frame, sparams, ϕparams, threads)
    return sample
end

@inline function _resize_line_samples!(sample :: LineSamples{3, TF, CuVector{TF}}, n :: Int) where {TF <: AbstractFloat}
    resize!(sample.grid, n)
    @inbounds for d in 1:3
        resize!(sample.origin[d], n)
        resize!(sample.direction[d], n)
    end
    return nothing
end

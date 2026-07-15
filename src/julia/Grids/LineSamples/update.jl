######################################################################################

# Resizable line-sample geometry updates.
#     by Wei-Shan Su,
#     July 16, 2026

######################################################################################

"""
    update!(sample, Cartesian, ParallelBeam, frame, xparams, yparams)
    update!(sample, Polar, ParallelBeam, frame, sparams, ϕparams)

Resize and rebuild a CPU-backed `LineSamples` using the same geometric inputs
as its constructors. Existing arrays are resized in place, origins and
directions are regenerated, and `sample.grid` is reset to zero.

# Parameters
- `sample`: Reusable three-dimensional line-sample storage.
- `Cartesian`, `Polar`: Coordinate-system dispatch tag.
- `ParallelBeam`: Parallel-beam model dispatch tag.
- `frame`: Frame defining the new origins and line direction.
- `xparams`, `yparams`: Cartesian plane axis specifications.
- `sparams`, `ϕparams`: Radial and angular plane axis specifications.

# Returns
- `LineSamples`: The same `sample` wrapper after resizing and rebuilding.
"""
function update!(sample :: LineSamples{3, TF, Vector{TF}}, coordinate_system :: Type{Cartesian}, beam_model :: Type{ParallelBeam}, frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    _resize!(sample, xparams[3] * yparams[3])
    build!(sample, coordinate_system, beam_model, frame, xparams, yparams)
    return sample
end

function update!(sample :: LineSamples{3, TF, Vector{TF}}, coordinate_system :: Type{Polar}, beam_model :: Type{ParallelBeam}, frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    _resize!(sample, sparams[3] * ϕparams[3])
    build!(sample, coordinate_system, beam_model, frame, sparams, ϕparams)
    return sample
end

@inline function _resize!(sample :: LineSamples{3, TF, Vector{TF}}, n :: Int) where {TF <: AbstractFloat}
    resize!(sample.grid, n)
    @inbounds for d in 1:3
        resize!(sample.origin[d], n)
        resize!(sample.direction[d], n)
    end
    return nothing
end

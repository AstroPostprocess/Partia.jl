######################################################################################

# Reusable line-sample geometry builders.
#     by Wei-Shan Su,
#     July 16, 2026

######################################################################################

"""
    build!(sample, Cartesian, ParallelBeam, frame, xparams, yparams)
    build!(sample, Polar, ParallelBeam, frame, sparams, ϕparams)

Rebuild a preallocated CPU-backed `LineSamples` using the same geometric
arguments as the corresponding constructor. Origins are regenerated from the
current frame, every direction is set to `frame_forward(frame)`, and all values
in `sample.grid` are reset to zero. Storage length must already match.

# Parameters
- `sample`: Preallocated three-dimensional line-sample storage.
- `Cartesian`, `Polar`: Coordinate-system dispatch tag.
- `ParallelBeam`: Parallel-beam model dispatch tag.
- `frame`: Frame defining the new origins and line direction.
- `xparams`, `yparams`: Cartesian plane axis specifications.
- `sparams`, `ϕparams`: Radial and angular plane axis specifications.

# Returns
- `nothing`: Geometry and values are updated in place.
"""
function build!(sample :: LineSamples{3, TF, Vector{TF}}, :: Type{Cartesian}, :: Type{ParallelBeam}, frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    expected = xparams[3] * yparams[3]
    length(sample) == expected || throw(DimensionMismatch("sample storage must have length $expected"))
    _cartesian_plane_coordinates!(sample.origin, frame, xparams, yparams)
    direction = frame_forward(frame)
    @inbounds for d in 1:3
        fill!(sample.direction[d], direction[d])
    end
    fill!(sample.grid, zero(TF))
    return nothing
end

function build!(sample :: LineSamples{3, TF, Vector{TF}}, :: Type{Polar}, :: Type{ParallelBeam}, frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    expected = sparams[3] * ϕparams[3]
    length(sample) == expected || throw(DimensionMismatch("sample storage must have length $expected"))
    _polar_plane_coordinates!(sample.origin, frame, sparams, ϕparams)
    direction = frame_forward(frame)
    @inbounds for d in 1:3
        fill!(sample.direction[d], direction[d])
    end
    fill!(sample.grid, zero(TF))
    return nothing
end

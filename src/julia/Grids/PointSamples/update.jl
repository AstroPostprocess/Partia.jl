######################################################################################

# Resizable point-sample geometry updates.
#     by Wei-Shan Su,
#     July 16, 2026

######################################################################################

"""
    update!(sample, Cartesian, frame, xparams, yparams[, zparams])
    update!(sample, Polar, frame, sparams, ϕparams)
    update!(sample, Cylindrical, frame, sparams, ϕparams, zparams)

Resize and rebuild a CPU-backed `PointSamples` using the same geometric inputs
as its constructors. Existing arrays are resized in place, the geometry is
regenerated from `frame`, and `sample.grid` is reset to zero.

# Parameters
- `sample`: Reusable three-dimensional point-sample storage.
- `Cartesian`, `Polar`, `Cylindrical`: Coordinate-system dispatch tag.
- `frame`: Frame defining the new position and orientation.
- `xparams`, `yparams`, `zparams`: Cartesian axis specifications.
- `sparams`, `ϕparams`: Radial and angular axis specifications.

# Returns
- `PointSamples`: The same `sample` wrapper after resizing and rebuilding.
"""
function update!(sample :: PointSamples{3, TF, Vector{TF}}, coordinate_system :: Type{Cartesian}, frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    _resize!(sample, xparams[3] * yparams[3])
    build!(sample, coordinate_system, frame, xparams, yparams)
    return sample
end

function update!(sample :: PointSamples{3, TF, Vector{TF}}, coordinate_system :: Type{Cartesian}, frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}, zparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    _resize!(sample, xparams[3] * yparams[3] * zparams[3])
    build!(sample, coordinate_system, frame, xparams, yparams, zparams)
    return sample
end

function update!(sample :: PointSamples{3, TF, Vector{TF}}, coordinate_system :: Type{Polar}, frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    _resize!(sample, sparams[3] * ϕparams[3])
    build!(sample, coordinate_system, frame, sparams, ϕparams)
    return sample
end

function update!(sample :: PointSamples{3, TF, Vector{TF}}, coordinate_system :: Type{Cylindrical}, frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}, zparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    _resize!(sample, sparams[3] * ϕparams[3] * zparams[3])
    build!(sample, coordinate_system, frame, sparams, ϕparams, zparams)
    return sample
end

@inline function _resize!(sample :: PointSamples{3, TF, Vector{TF}}, n :: Int) where {TF <: AbstractFloat}
    resize!(sample.grid, n)
    @inbounds for d in 1:3
        resize!(sample.coor[d], n)
    end
    return nothing
end

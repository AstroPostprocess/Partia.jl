######################################################################################

# Reusable point-sample geometry builders.
#     by Wei-Shan Su,
#     July 16, 2026

######################################################################################

"""
    build!(sample, Cartesian, frame, xparams, yparams)
    build!(sample, Cartesian, frame, xparams, yparams, zparams)
    build!(sample, Polar, frame, sparams, ϕparams)
    build!(sample, Cylindrical, frame, sparams, ϕparams, zparams)

Rebuild the geometry of a preallocated CPU-backed `PointSamples` using the same
geometric arguments as the corresponding constructor. The coordinate arrays
must already have the required length. Coordinates are overwritten from the
current `frame`, and all values in `sample.grid` are reset to zero.

# Parameters
- `sample`: Preallocated three-dimensional point-sample storage.
- `Cartesian`, `Polar`, `Cylindrical`: Coordinate-system dispatch tag.
- `frame`: Frame defining the new position and orientation.
- `xparams`, `yparams`, `zparams`: Cartesian axis specifications.
- `sparams`, `ϕparams`: Radial and angular axis specifications.

# Returns
- `nothing`: Geometry and values are updated in place.
"""
function build!(sample :: PointSamples{3, TF, Vector{TF}}, :: Type{Cartesian}, frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    expected = xparams[3] * yparams[3]
    length(sample) == expected || throw(DimensionMismatch("sample storage must have length $expected"))
    _cartesian_plane_coordinates!(sample.coor, frame, xparams, yparams)
    fill!(sample.grid, zero(TF))
    return nothing
end

function build!(sample :: PointSamples{3, TF, Vector{TF}}, :: Type{Cartesian}, frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}, zparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    expected = xparams[3] * yparams[3] * zparams[3]
    length(sample) == expected || throw(DimensionMismatch("sample storage must have length $expected"))
    _cartesian_box_coordinates!(sample.coor, frame, xparams, yparams, zparams)
    fill!(sample.grid, zero(TF))
    return nothing
end

function build!(sample :: PointSamples{3, TF, Vector{TF}}, :: Type{Polar}, frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    expected = sparams[3] * ϕparams[3]
    length(sample) == expected || throw(DimensionMismatch("sample storage must have length $expected"))
    _polar_plane_coordinates!(sample.coor, frame, sparams, ϕparams)
    fill!(sample.grid, zero(TF))
    return nothing
end

function build!(sample :: PointSamples{3, TF, Vector{TF}}, :: Type{Cylindrical}, frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}, zparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    expected = sparams[3] * ϕparams[3] * zparams[3]
    length(sample) == expected || throw(DimensionMismatch("sample storage must have length $expected"))
    _cylindrical_coordinates!(sample.coor, frame, sparams, ϕparams, zparams)
    fill!(sample.grid, zero(TF))
    return nothing
end

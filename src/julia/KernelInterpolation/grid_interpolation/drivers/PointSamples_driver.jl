######################################################################################

#     PointSamples interpolation drivers

######################################################################################

#     Gather interpolation

######################################################################################
"""
    PointSamples_interpolation!(
        grids :: NTuple{L, PS},
        input :: AbstractInterpolationInput{3, TF, Vector{TF}},
        catalog_consice :: InterpolationCatalogConcise{3, N, G, Div, C},
        LBVH :: LinearBVH{3, TF, Vector{TF}},
        :: Type{itpGather},
    ) where {N, G, Div, C, L, TF <: AbstractFloat, PS <: PointSamples{3, TF, Vector{TF}}}

Evaluate point-sample interpolation in place using gather interpolation and
already prepared interpolation state.

This is the core CPU implementation for reusable output grids. It assumes the
output `grids`, particle `input`, concise catalog, and Morton-reordered `LBVH`
have already been prepared by an outer wrapper. The function writes directly
into the supplied `PointSamples` grids and does not allocate a `GridBundle`.
"""
function PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{3, N, G, Div, C}, LBVH :: LinearBVH{3, TF, Vector{TF}}, :: Type{itpGather}) where {N, G, Div, C, L, TF <: AbstractFloat, PS <: PointSamples{3, TF, Vector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, Vector{TF}}}
    # Exit if nothing to do
    L == 0 && return nothing

    # Make sure every grids share the same geometry
    if L > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output PointSamples grids must share the same point coordinates. " *
            "Expected every grid to reuse the same coordinate vectors."
        ))
    end

    # Prepare multiprocessing
    npoints = length(grids[1])

    # Do interpolation
    @inbounds @threads for i in 1:npoints
        _point_samples_interpolation_kernel!(grids, i, input, catalog_consice, LBVH, itpGather)
    end

    return nothing
end

######################################################################################

#     Scatter interpolation

######################################################################################
"""
    PointSamples_interpolation!(
        grids :: NTuple{L, PS},
        input :: AbstractInterpolationInput{3, TF, Vector{TF}},
        catalog_consice :: InterpolationCatalogConcise{3, N, G, Div, C},
        LBVH :: LinearBVH{3, TF, Vector{TF}},
        :: Type{itpScatter},
    ) where {N, G, Div, C, L, TF <: AbstractFloat, PS <: PointSamples{3, TF, Vector{TF}}}

Evaluate point-sample interpolation in place using scatter interpolation and
already prepared interpolation state.

This is the core CPU implementation for reusable output grids. It assumes the
output `grids`, particle `input`, concise catalog, and Morton-reordered `LBVH`
have already been prepared by an outer wrapper. The function writes directly
into the supplied `PointSamples` grids and does not allocate a `GridBundle`.
"""
function PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{3, N, G, Div, C}, LBVH :: LinearBVH{3, TF, Vector{TF}}, :: Type{itpScatter}) where {N, G, Div, C, L, TF <: AbstractFloat, PS <: PointSamples{3, TF, Vector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, Vector{TF}}}
    # Exit if nothing to do
    L == 0 && return nothing

    # Make sure every grids share the same geometry
    if L > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output PointSamples grids must share the same point coordinates. " *
            "Expected every grid to reuse the same coordinate vectors."
        ))
    end

    # Prepare multiprocessing
    npoints = length(grids[1])

    # Do interpolation
    @inbounds @threads for i in 1:npoints
        _point_samples_interpolation_kernel!(grids, i, input, catalog_consice, LBVH, itpScatter)
    end

    return nothing
end

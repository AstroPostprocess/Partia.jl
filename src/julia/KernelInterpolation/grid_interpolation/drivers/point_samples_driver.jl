######################################################################################

#     PointSamples interpolation drivers

######################################################################################

#     Gather interpolation

######################################################################################
"""
    PointSamples_interpolation_prepared!(
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

# Parameters
- `grids`: Preallocated point-sample output grids.
- `input`: Prepared particle-side interpolation input.
- `catalog_consice`: Concise execution catalog.
- `LBVH`: Prebuilt hierarchy matching the current input order.
- `::Type{itpGather}`: Selects gather interpolation.

# Returns
- `nothing`: The supplied grids are updated in place.
"""
function PointSamples_interpolation_prepared!(grids :: NTuple{L, PS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{3, N, G, Div, C}, LBVH :: LinearBVH{3, TF, Vector{TF}}, :: Type{itpGather}) where {N, G, Div, C, L, TF <: AbstractFloat, PS <: PointSamples{3, TF, Vector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, Vector{TF}}}
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

"""
    PointSamples_interpolation_prepared!(
        grids :: NTuple{L, PS},
        input :: AbstractInterpolationInput{2, TF, Vector{TF}},
        catalog_consice :: InterpolationCatalogConcise{2, N, G, Div, 0},
        LBVH :: LinearBVH{2, TF, Vector{TF}},
        :: Type{itpGather},
    )

Evaluate two-dimensional point-sample interpolation in place using gather
interpolation and already prepared interpolation state. The concise catalog is
restricted to zero curl requests.

# Parameters
- `grids`: Preallocated point-sample output grids.
- `input`: Prepared particle-side interpolation input.
- `catalog_consice`: Concise execution catalog.
- `LBVH`: Prebuilt hierarchy matching the current input order.
- `::Type{itpGather}`: Selects gather interpolation.

# Returns
- `nothing`: The supplied grids are updated in place.
"""
function PointSamples_interpolation_prepared!(grids :: NTuple{L, PS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{2, N, G, Div, 0}, LBVH :: LinearBVH{2, TF, Vector{TF}}, :: Type{itpGather}) where {N, G, Div, L, TF <: AbstractFloat, PS <: PointSamples{2, TF, Vector{TF}}, INPUT <: AbstractInterpolationInput{2, TF, Vector{TF}}}
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
    PointSamples_interpolation_prepared!(
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

# Parameters
- `grids`: Preallocated point-sample output grids.
- `input`: Prepared particle-side interpolation input.
- `catalog_consice`: Concise execution catalog.
- `LBVH`: Prebuilt hierarchy matching the current input order.
- `::Type{itpScatter}`: Selects scatter interpolation.

# Returns
- `nothing`: The supplied grids are updated in place.
"""
function PointSamples_interpolation_prepared!(grids :: NTuple{L, PS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{3, N, G, Div, C}, LBVH :: LinearBVH{3, TF, Vector{TF}}, :: Type{itpScatter}) where {N, G, Div, C, L, TF <: AbstractFloat, PS <: PointSamples{3, TF, Vector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, Vector{TF}}}
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

"""
    PointSamples_interpolation_prepared!(
        grids :: NTuple{L, PS},
        input :: AbstractInterpolationInput{2, TF, Vector{TF}},
        catalog_consice :: InterpolationCatalogConcise{2, N, G, Div, 0},
        LBVH :: LinearBVH{2, TF, Vector{TF}},
        :: Type{itpScatter},
    )

Evaluate two-dimensional point-sample interpolation in place using scatter
interpolation and already prepared interpolation state. The concise catalog is
restricted to zero curl requests.

# Parameters
- `grids`: Preallocated point-sample output grids.
- `input`: Prepared particle-side interpolation input.
- `catalog_consice`: Concise execution catalog.
- `LBVH`: Prebuilt hierarchy matching the current input order.
- `::Type{itpScatter}`: Selects scatter interpolation.

# Returns
- `nothing`: The supplied grids are updated in place.
"""
function PointSamples_interpolation_prepared!(grids :: NTuple{L, PS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{2, N, G, Div, 0}, LBVH :: LinearBVH{2, TF, Vector{TF}}, :: Type{itpScatter}) where {N, G, Div, L, TF <: AbstractFloat, PS <: PointSamples{2, TF, Vector{TF}}, INPUT <: AbstractInterpolationInput{2, TF, Vector{TF}}}
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

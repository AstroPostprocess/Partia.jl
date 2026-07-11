######################################################################################

#     LineSamples interpolation drivers

######################################################################################
"""
    LineSamples_interpolation_prepared!(
        grids :: NTuple{N, LS},
        input :: AbstractInterpolationInput{3, TF, Vector{TF}},
        catalog_consice :: InterpolationCatalogConcise{3, N, 0, 0, 0},
        LBVH :: LinearBVH{3, TF, Vector{TF}},
    ) where {N, TF <: AbstractFloat, LS <: LineSamples{3, TF, Vector{TF}}}

Evaluate line-sample interpolation in place using already prepared interpolation
state.

This is the core CPU implementation for reusable output grids. It assumes the
output `grids`, particle `input`, concise catalog, and Morton-reordered `LBVH`
have already been prepared by an outer wrapper. The function writes directly
into the supplied `LineSamples` grids and does not allocate a `GridBundle`.
"""
function LineSamples_interpolation_prepared!(grids :: NTuple{N, LS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{3, N, 0, 0, 0}, LBVH :: LinearBVH{3, TF, Vector{TF}}) where {N, TF <: AbstractFloat, LS <: LineSamples{3, TF, Vector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, Vector{TF}}}
    # Exit if nothing to do
    N == 0 && return nothing

    # Make sure every grids share the same geometry
    if N > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output LineSamples grids must share the same line geometry. " *
            "Expected every grid to reuse the same origin and direction vectors."
        ))
    end

    # Prepare multiprocessing
    npoints = length(grids[1])

    # Do interpolation
    @inbounds @threads for i in 1:npoints
        _line_samples_interpolation_kernel!(grids, i, input, catalog_consice, LBVH)
    end

    return nothing
end

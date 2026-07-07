######################################################################################

#     Metal PointSamples interpolation drivers

######################################################################################

#     Gather interpolation

######################################################################################
function Partia.PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{3, N, G, Div, C}, LBVH :: LinearBVH{3, TF, MtlVector{TF}}, :: Type{itpGather}) where {N, G, Div, C, L, TF <: Float32, PS <: PointSamples{3, TF, MtlVector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, MtlVector{TF}}}
    L == 0 && return nothing

    if L > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output PointSamples grids must share the same point coordinates. " *
            "Expected every grid to reuse the same coordinate vectors."
        ))
    end

    npoints = length(grids[1])
    npoints == 0 && return nothing

    @metal threads=(256,) groups=(cld(npoints, 256)) _point_samples_interpolation_kernel!(grids, input, catalog_consice, LBVH, itpGather)
    Metal.synchronize()

    return nothing
end

######################################################################################

#     Scatter interpolation

######################################################################################
function Partia.PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{3, N, G, Div, C}, LBVH :: LinearBVH{3, TF, MtlVector{TF}}, :: Type{itpScatter}) where {N, G, Div, C, L, TF <: Float32, PS <: PointSamples{3, TF, MtlVector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, MtlVector{TF}}}
    L == 0 && return nothing

    if L > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output PointSamples grids must share the same point coordinates. " *
            "Expected every grid to reuse the same coordinate vectors."
        ))
    end

    npoints = length(grids[1])
    npoints == 0 && return nothing

    @metal threads=(256,) groups=(cld(npoints, 256)) _point_samples_interpolation_kernel!(grids, input, catalog_consice, LBVH, itpScatter)
    Metal.synchronize()

    return nothing
end

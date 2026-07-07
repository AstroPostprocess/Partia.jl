######################################################################################

#     Metal LineSamples interpolation drivers

######################################################################################
function Partia.LineSamples_interpolation_prepared!(grids :: NTuple{N, LS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{3, N, 0, 0, 0}, LBVH :: LinearBVH{3, TF, MtlVector{TF}}) where {N, TF <: Float32, LS <: LineSamples{3, TF, MtlVector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, MtlVector{TF}}}
    N == 0 && return nothing

    if N > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output LineSamples grids must share the same line geometry. " *
            "Expected every grid to reuse the same origin and direction vectors."
        ))
    end

    npoints = length(grids[1])
    npoints == 0 && return nothing

    tables_Mtl = _line_integrated_tables_Mtl()
    @metal threads=(256,) groups=(cld(npoints, 256)) _line_samples_interpolation_kernel!(grids, input, catalog_consice, LBVH, tables_Mtl, itpScatter)
    Metal.synchronize()

    return nothing
end

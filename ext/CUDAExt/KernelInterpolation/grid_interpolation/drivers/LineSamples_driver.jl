######################################################################################

#     CUDA LineSamples interpolation drivers

######################################################################################
function Partia.LineSamples_interpolation_prepared!(grids :: NTuple{N, LS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{3, N, 0, 0, 0}, LBVH :: LinearBVH{3, TF, CuVector{TF}}) where {N, TF <: AbstractFloat, LS <: LineSamples{3, TF, CuVector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}}
    N == 0 && return nothing

    if N > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output LineSamples grids must share the same line geometry. " *
            "Expected every grid to reuse the same origin and direction vectors."
        ))
    end

    npoints = length(grids[1])
    npoints == 0 && return nothing

    @cuda threads=(256,) blocks=(cld(npoints, 256)) _line_samples_interpolation_kernel!(grids, input, catalog_consice, LBVH, itpScatter)
    CUDA.synchronize()

    return nothing
end

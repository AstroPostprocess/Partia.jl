######################################################################################

#     CUDA LineSamples interpolation drivers

######################################################################################
"""
    LineSamples_interpolation!(grids, input, catalog_consice, LBVH,
                                        ::Val{ThreadsPerBlock}=Val(256))

Launch the prepared CUDA line-sample interpolation kernel.

# Parameters
- `grids`: Preallocated CUDA line-sample output grids.
- `input`: Prepared CUDA interpolation input.
- `catalog_consice`: Concise scalar execution catalog.
- `LBVH`: Prebuilt hierarchy matching `input`.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `nothing`: The supplied grids are updated in place.
"""
function Partia.LineSamples_interpolation!(grids :: NTuple{N, LS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{3, N, 0, 0, 0}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, :: Val{ThreadsPerBlock} = Val(256)) where {N, ThreadsPerBlock, TF <: AbstractFloat, LS <: LineSamples{3, TF, CuVector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}}
    N == 0 && return nothing

    if N > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output LineSamples grids must share the same line geometry. " *
            "Expected every grid to reuse the same origin and direction vectors."
        ))
    end

    npoints = length(grids[1])
    npoints == 0 && return nothing

    @cuda threads=(ThreadsPerBlock,) blocks=(cld(npoints, ThreadsPerBlock)) _line_samples_interpolation_kernel!(grids, input, catalog_consice, LBVH, itpScatter)
    CUDA.synchronize()

    return nothing
end

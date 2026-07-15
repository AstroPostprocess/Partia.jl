######################################################################################

#     Metal LineSamples interpolation drivers

######################################################################################
"""
    LineSamples_interpolation_prepared!(grids, input, catalog_consice, LBVH,
                                        ::Val{ThreadsPerGroup}=Val(256))

Launch the prepared Metal line-sample interpolation kernel.

# Parameters
- `grids`: Preallocated Metal line-sample output grids.
- `input`: Prepared Metal interpolation input.
- `catalog_consice`: Concise scalar execution catalog.
- `LBVH`: Prebuilt hierarchy matching `input`.
- `::Val{ThreadsPerGroup}`: Metal threads per threadgroup.

# Returns
- `nothing`: The supplied grids are updated in place.
"""
function Partia.LineSamples_interpolation_prepared!(grids :: NTuple{N, LS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{3, N, 0, 0, 0}, LBVH :: LinearBVH{3, TF, MtlVector{TF}}, :: Val{ThreadsPerGroup} = Val(256)) where {N, ThreadsPerGroup, TF <: Float32, LS <: LineSamples{3, TF, MtlVector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, MtlVector{TF}}}
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
    @metal threads=(ThreadsPerGroup,) groups=(cld(npoints, ThreadsPerGroup)) _line_samples_interpolation_kernel!(grids, input, catalog_consice, LBVH, tables_Mtl, itpScatter)
    Metal.synchronize()

    return nothing
end

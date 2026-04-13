"""
    LineSamples_interpolation(backend::MetalComputeBackend, grid_template::LineSamples{3,TF},
                              input::InterpolationInput{3,TF}, catalog::InterpolationCatalog{3, N, 0, 0, 0, N},
                              itp_strategy::Type{ITPSTRATEGY}=itpScatter)

Performs line-integrated SPH interpolation on the GPU using Metal.

This routine mirrors the CPU `LineSamples_interpolation` path: it only supports
scalar line-integrated quantities and only the `itpScatter` strategy.
"""
function Partia.LineSamples_interpolation(::MetalComputeBackend, grid_template::LineSamples{3, TF}, input::InterpolationInput{3, TF}, catalog::InterpolationCatalog{3, N, 0, 0, 0, N}, itp_strategy::Type{ITPSTRATEGY} = itpScatter) where {N, TF <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    itp_strategy === itpScatter || throw(ArgumentError(
        "LineSamples_interpolation only supports itpScatter. " *
        "Line-integrated samples do not have a well-defined query smoothing length ha, " *
        "so itpGather and itpSymmetric are not supported."
    ))

    grids, LBVH, names, catalog_consice = Partia.initialize_interpolation(Partia.CPUComputeBackend(), grid_template, input, catalog)
    @info "     SPH Interpolation: Copying interpolated grids to device memory..."
    input_Mtl = to_MtlVector(input)
    grids_Mtl = ntuple(i -> to_MtlVector(grids[i]), Val(N))
    LBVH_Mtl = to_MtlVector(LBVH)
    tables_Mtl = _line_integrated_tables_Mtl()
    @info "     SPH Interpolation: End copying interpolated grids to device memory."

    npoints = length(grid_template)
    @info "     SPH Interpolation: Start interpolation..."
    @metal threads=(256,) groups=(cld(npoints, 256)) _line_samples_interpolation_kernel!(grids_Mtl, input_Mtl, catalog_consice, LBVH_Mtl, tables_Mtl, itpScatter)
    Metal.synchronize()
    @info "     SPH Interpolation: End interpolation."
    @info "     SPH Interpolation: Copying interpolated grids back to host memory..."
    grids_result = ntuple(i -> Partia.to_HostVector(grids_Mtl[i]), Val(N))
    @info "     SPH Interpolation: End copying interpolated grids back to host memory."
    return GridBundle(grids_result, names)
end

"""
    LineSamples_interpolation(backend::MetalComputeBackend, grid_template::LineSamples{3,TF},
                              input::InterpolationInput{3,TF}, LBVH::LinearBVH{3,TF},
                              catalog::InterpolationCatalog{3, N, 0, 0, 0, N},
                              itp_strategy::Type{ITPSTRATEGY}=itpScatter)

Performs line-integrated SPH interpolation on the GPU using Metal with an
externally supplied `LinearBVH`.

This routine mirrors the existing Metal line-sample path, but skips LBVH
construction and instead reuses the provided hierarchy after checking that its
leaf order matches the current ordering of `input`.
"""
function Partia.LineSamples_interpolation(::MetalComputeBackend, grid_template::LineSamples{3, TF}, input::InterpolationInput{3, TF}, LBVH::LinearBVH{3, TF}, catalog::InterpolationCatalog{3, N, 0, 0, 0, N}, itp_strategy::Type{ITPSTRATEGY} = itpScatter) where {N, TF <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    itp_strategy === itpScatter || throw(ArgumentError(
        "LineSamples_interpolation only supports itpScatter. " *
        "Line-integrated samples do not have a well-defined query smoothing length ha, " *
        "so itpGather and itpSymmetric are not supported."
    ))

    Partia.matches_lbvh_leaf_order(input, LBVH) || throw(ArgumentError(
        "Provided LBVH leaf order does not match the current input ordering. " *
        "Ensure the LBVH was built from the same Morton-reordered input."
    ))

    grids, names, catalog_consice = Partia.initialize_interpolation(Partia.CPUComputeBackend(), grid_template, catalog)
    @info "     SPH Interpolation: Copying interpolated grids to device memory..."
    input_Mtl = to_MtlVector(input)
    grids_Mtl = ntuple(i -> to_MtlVector(grids[i]), Val(N))
    LBVH_Mtl = to_MtlVector(LBVH)
    tables_Mtl = _line_integrated_tables_Mtl()
    @info "     SPH Interpolation: End copying interpolated grids to device memory."

    npoints = length(grid_template)
    @info "     SPH Interpolation: Start interpolation..."
    @metal threads=(256,) groups=(cld(npoints, 256)) _line_samples_interpolation_kernel!(grids_Mtl, input_Mtl, catalog_consice, LBVH_Mtl, tables_Mtl, itpScatter)
    Metal.synchronize()
    @info "     SPH Interpolation: End interpolation."
    @info "     SPH Interpolation: Copying interpolated grids back to host memory..."
    grids_result = ntuple(i -> Partia.to_HostVector(grids_Mtl[i]), Val(N))
    @info "     SPH Interpolation: End copying interpolated grids back to host memory."
    return GridBundle(grids_result, names)
end

"""
    PointSamples_interpolation(backend::MetalComputeBackend, grid_template::PointSamples{3,TF},
                          input::InterpolationInput{3,TF}, catalog::InterpolationCatalog{3, N, G, Div, C, L},
                          itp_strategy::Type{ITPSTRATEGY} = itpSymmetric)

Performs SPH grid interpolation on the GPU using Apple's Metal backend.

This routine mirrors the CPU `PointSamples_interpolation` path: it supports the
same interpolation catalog and strategy combinations while offloading the point
evaluation kernel to the GPU.
"""
function Partia.PointSamples_interpolation(::MetalComputeBackend, grid_template::PointSamples{3, TF}, input::InterpolationInput{3, TF}, catalog::InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) where {N, G, Div, C, L, TF <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids, LBVH, names, catalog_consice = Partia.initialize_interpolation(Partia.CPUComputeBackend(), grid_template, input, catalog)
    @info "     SPH Interpolation: Copying interpolated grids to device memory..."
    input_Mtl = to_MtlVector(input)
    grids_Mtl = ntuple(i -> to_MtlVector(grids[i]), Val(L))
    LBVH_Mtl = to_MtlVector(LBVH)
    @info "     SPH Interpolation: End copying interpolated grids to device memory."

    npoints = length(grid_template)
    @info "     SPH Interpolation: Start interpolation..."
    @metal threads=(256,) groups=(cld(npoints, 256)) _point_samples_interpolation_kernel!(grids_Mtl, input_Mtl, catalog_consice, LBVH_Mtl, itp_strategy)
    Metal.synchronize()
    @info "     SPH Interpolation: End interpolation."
    @info "     SPH Interpolation: Copying interpolated grids back to host memory..."
    grids_result = ntuple(i -> Partia.to_HostVector(grids_Mtl[i]), Val(L))
    @info "     SPH Interpolation: End copying interpolated grids back to host memory."
    return GridBundle(grids_result, names)
end

"""
    PointSamples_interpolation(backend::MetalComputeBackend, grid_template::PointSamples{3,TF},
                               input::InterpolationInput{3,TF}, LBVH::LinearBVH{3,TF},
                               catalog::InterpolationCatalog{3, N, G, Div, C, L},
                               itp_strategy::Type{ITPSTRATEGY} = itpSymmetric)

Performs SPH grid interpolation on the GPU using Apple's Metal backend with an
externally supplied `LinearBVH`.

This routine mirrors the existing Metal point-sample path, but skips LBVH
construction and instead reuses the provided hierarchy after checking that its
leaf order matches the current ordering of `input`.
"""
function Partia.PointSamples_interpolation(::MetalComputeBackend, grid_template::PointSamples{3, TF}, input::InterpolationInput{3, TF}, LBVH::LinearBVH{3, TF}, catalog::InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) where {N, G, Div, C, L, TF <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    Partia.matches_lbvh_leaf_order(input, LBVH) || throw(ArgumentError(
        "Provided LBVH leaf order does not match the current input ordering. " *
        "Ensure the LBVH was built from the same Morton-reordered input."
    ))

    grids, names, catalog_consice = Partia.initialize_interpolation(Partia.CPUComputeBackend(), grid_template, catalog)
    @info "     SPH Interpolation: Copying interpolated grids to device memory..."
    input_Mtl = to_MtlVector(input)
    grids_Mtl = ntuple(i -> to_MtlVector(grids[i]), Val(L))
    LBVH_Mtl = to_MtlVector(LBVH)
    @info "     SPH Interpolation: End copying interpolated grids to device memory."

    npoints = length(grid_template)
    @info "     SPH Interpolation: Start interpolation..."
    @metal threads=(256,) groups=(cld(npoints, 256)) _point_samples_interpolation_kernel!(grids_Mtl, input_Mtl, catalog_consice, LBVH_Mtl, itp_strategy)
    Metal.synchronize()
    @info "     SPH Interpolation: End interpolation."
    @info "     SPH Interpolation: Copying interpolated grids back to host memory..."
    grids_result = ntuple(i -> Partia.to_HostVector(grids_Mtl[i]), Val(L))
    @info "     SPH Interpolation: End copying interpolated grids back to host memory."
    return GridBundle(grids_result, names)
end

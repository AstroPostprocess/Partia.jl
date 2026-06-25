"""
    PointSamples_interpolation(backend :: MetalComputeBackend, grid_template :: PointSamples{3, TF},
                               input :: AbstractInterpolationInput{3, TF}, catalog :: InterpolationCatalog{3, N, G, Div, C, L},
                               itp_strategy :: Type{ITPSTRATEGY} = itpScatter)

Perform SPH interpolation over an arbitrary point-sample grid using Metal execution.
This routine prepares the interpolation structures on the CPU, copies particle
input, output grids, and the LBVH to Metal device memory, evaluates each grid
point on the GPU, and copies the interpolated grids back to host memory.

# Parameters
- `backend :: MetalComputeBackend`
  Execution backend specifying Metal-based interpolation.

- `grid_template :: PointSamples{3, TF}`
  Template grid defining dimensionality, coordinate arrays, and memory layout of
  all output grids.

- `input :: AbstractInterpolationInput{3, TF}`
  The interpolation input holding particle positions, smoothing lengths, field
  data, and the SPH kernel.

- `catalog :: InterpolationCatalog{3, N, G, Div, C, L}`
  Full interpolation catalog describing which scalar, gradient, divergence, and
  curl quantities are to be produced.

- `itp_strategy :: Type{ITPSTRATEGY}`
  Interpolation strategy type controlling gather/scatter modes.

# Returns
`GridBundle{L, typeof(grids[1])}` containing:
- `grids` : NTuple of output grids storing interpolated results.
- `names` : Ordered list of all output quantity names, matching the grid tuple order.
"""
function Partia.PointSamples_interpolation( :: MetalComputeBackend, grid_template :: PointSamples{3, TF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, C, L, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
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
    PointSamples_interpolation(backend :: MetalComputeBackend, grid_template :: PointSamples{3, TF},
                               input :: AbstractInterpolationInput{3, TF}, LBVH :: LinearBVH{3, TF},
                               catalog :: InterpolationCatalog{3, N, G, Div, C, L},
                               itp_strategy :: Type{ITPSTRATEGY} = itpScatter)

Perform SPH interpolation over an arbitrary point-sample grid using Metal
execution with an externally supplied `LinearBVH`.

This routine assumes that `input` has already been reordered into the same
Morton leaf order used to build `LBVH`. Before interpolation begins, it checks
that the reordered spatial layout of `input` matches the leaf ordering stored in
`LBVH`. It then allocates output grids, builds the concise interpolation
catalog, copies the required data to Metal device memory, evaluates each grid
point on the GPU, and copies the interpolated grids back to host memory.

# Parameters
- `backend :: MetalComputeBackend`
  Execution backend specifying Metal-based interpolation.

- `grid_template :: PointSamples{3, TF}`
  Template grid defining dimensionality, coordinate arrays, and memory layout of
  all output grids.

- `input :: AbstractInterpolationInput{3, TF}`
  The interpolation input holding particle positions, smoothing lengths, field
  data, and the SPH kernel. Its current ordering must already match the LBVH
  leaf ordering.

- `LBVH :: LinearBVH{3, TF}`
  A prebuilt `LinearBVH` used for neighbour traversal during interpolation.

- `catalog :: InterpolationCatalog{3, N, G, Div, C, L}`
  Full interpolation catalog describing which scalar, gradient, divergence, and
  curl quantities are to be produced.

- `itp_strategy :: Type{ITPSTRATEGY}`
  Interpolation strategy type controlling gather/scatter modes.

# Returns
`GridBundle{L, typeof(grids[1])}` containing:
- `grids` : NTuple of output grids storing interpolated results.
- `names` : Ordered list of all output quantity names, matching the grid tuple order.

# Throws
- `ArgumentError`: If the leaf order stored in `LBVH` does not match the
  current spatial ordering of `input`.
"""
function Partia.PointSamples_interpolation( :: MetalComputeBackend, grid_template :: PointSamples{3, TF}, input :: INPUT, LBVH :: LinearBVH{3, TF}, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, C, L, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
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

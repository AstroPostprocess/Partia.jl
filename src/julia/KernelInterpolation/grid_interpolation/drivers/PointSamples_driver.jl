"""
    PointSamples_interpolation(backend :: CPUComputeBackend, grid_template :: PointSamples{D},
                          input :: InterpolationInput{3,TF}, catalog :: InterpolationCatalog{3, N, G, Div, C, L},
                          itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric)

Performs SPH interpolation over an arbitrary grid using CPU execution.
This routine dispatches to the CPU backend, prepares all interpolation structures,
and evaluates each grid point in parallel using threaded execution.

# Parameters
- `backend :: CPUComputeBackend`
  Execution backend specifying CPU-based interpolation.

- `grid_template :: PointSamples{D}`
  Template grid defining dimensionality, coordinate arrays, and memory layout of
  all output grids.

- `input :: InterpolationInput{3,TF}`
  The `InterpolationInput` holding particle positions, smoothing lengths, field
  data, and the SPH kernel.

- `catalog :: InterpolationCatalog{3, N, G, Div, C, L}`
  Full interpolation catalog describing which scalar, gradient, divergence, and
  curl quantities are to be produced.

- `itp_strategy :: Type{ITPSTRATEGY}`
  Interpolation strategy type controlling symmetric/gather/scatter modes.

# Returns
`GridBundle{L, typeof(grids[1])}` containing:
- `grids` — NTuple of output grids storing interpolated results.
- `names` — Ordered list of all output quantity names, matching the grid tuple order.
"""
function PointSamples_interpolation(backend :: CPUComputeBackend, grid_template :: PointSamples{3, TF}, input :: InterpolationInput{3, TF}, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {N, G, Div, C, L, TF <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids_result, LBVH, names, catalog_consice = initialize_interpolation(backend, grid_template, input, catalog)
    npoints = length(grid_template)
    @info "     SPH Interpolation: Start interpolation..."
    @inbounds @threads for i in 1:npoints
        # Do single point interpolation
        _point_samples_interpolation_kernel!(backend, grids_result, i, input, catalog_consice, LBVH, itp_strategy)

    end
    @info "     SPH Interpolation: End interpolation..."

    # Output (No extra operation, keep interface clean)
    grids = grids_result
    return GridBundle(grids, names)
end


"""
    PointSamples_interpolation(backend :: CPUComputeBackend, grid_template :: PointSamples{D},
                               input :: InterpolationInput{3,TF}, LBVH :: LinearBVH{3,TF},
                               catalog :: InterpolationCatalog{3, N, G, Div, C, L},
                               itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric)

Perform SPH interpolation over an arbitrary point-sample grid using CPU
execution with an externally supplied `LinearBVH`.

This routine assumes that `input` has already been reordered into the same
Morton leaf order used to build `LBVH`. Before interpolation begins, it checks
that the reordered spatial layout of `input` matches the leaf ordering stored in
`LBVH`. It then allocates output grids, builds the concise interpolation
catalog, and evaluates each grid point in parallel using threaded execution.

# Parameters
- `backend :: CPUComputeBackend`
  Execution backend specifying CPU-based interpolation.

- `grid_template :: PointSamples{D}`
  Template grid defining dimensionality, coordinate arrays, and memory layout of
  all output grids.

- `input :: InterpolationInput{3,TF}`
  The `InterpolationInput` holding particle positions, smoothing lengths, field
  data, and the SPH kernel. Its current ordering must already match the LBVH
  leaf ordering.

- `LBVH :: LinearBVH{3,TF}`
  A prebuilt `LinearBVH` used for neighbour traversal during interpolation.

- `catalog :: InterpolationCatalog{3, N, G, Div, C, L}`
  Full interpolation catalog describing which scalar, gradient, divergence, and
  curl quantities are to be produced.

- `itp_strategy :: Type{ITPSTRATEGY}`
  Interpolation strategy type controlling symmetric/gather/scatter modes.

# Returns
`GridBundle{L, typeof(grids[1])}` containing:
- `grids` : NTuple of output grids storing interpolated results.
- `names` : Ordered list of all output quantity names, matching the grid tuple order.

# Throws
- `ArgumentError`: If the leaf order stored in `LBVH` does not match the
  current spatial ordering of `input`.
"""
function PointSamples_interpolation(backend :: CPUComputeBackend, grid_template :: PointSamples{3, TF}, input :: InterpolationInput{3, TF}, LBVH :: LinearBVH{3, TF}, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {N, G, Div, C, L, TF <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Consistency test for LBVH
    matches_lbvh_leaf_order(input, LBVH) || throw(ArgumentError(
        "Provided LBVH leaf order does not match the current input ordering. " *
        "Ensure the LBVH was built from the same Morton-reordered input."
    ))

    grids_result, names, catalog_consice = initialize_interpolation(backend, grid_template, catalog)
    npoints = length(grid_template)
    @info "     SPH Interpolation: Start interpolation..."
    @inbounds @threads for i in 1:npoints
        # Do single point interpolation
        _point_samples_interpolation_kernel!(backend, grids_result, i, input, catalog_consice, LBVH, itp_strategy)

    end
    @info "     SPH Interpolation: End interpolation..."

    # Output (No extra operation, keep interface clean)
    grids = grids_result
    return GridBundle(grids, names)
end

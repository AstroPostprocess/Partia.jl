"""
    LineSamples_interpolation(backend :: MetalComputeBackend, grid_template :: LineSamples{3, TF},
                              input :: AbstractInterpolationInput{3, TF}, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N},
                              itp_strategy :: Type{ITPSTRATEGY} = itpScatter)

Perform Metal-based SPH interpolation for an unstructured collection of line samples.
This routine prepares the interpolation structures on the CPU, copies particle
input, output grids, the LBVH, and line-integrated kernel data to Metal device
memory, evaluates each line sample on the GPU, and copies the interpolated grids
back to host memory.

Each sample is interpreted as a line primitive defined by the corresponding
origin and direction stored in `grid_template`. For the `i`-th sample, the
interpolation kernel evaluates line-integrated quantities associated with that
line and stores the resulting scalar values into the output grids.

At present, this routine only supports `itpScatter`. For line-integrated
samples there is no well-defined query smoothing length `ha`, so
`itpGather` is rejected explicitly.

# Parameters
- `backend :: MetalComputeBackend`
  Execution backend specifying Metal-based interpolation.

- `grid_template :: LineSamples{3, TF}`
  Template sample container defining the dimensionality, line geometry, and
  output container layout.

- `input :: AbstractInterpolationInput{3, TF}`
  An interpolation input object containing particle positions, smoothing
  lengths, field values, and the SPH kernel.

- `catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}`
  Interpolation catalog describing the requested output quantities.
  In the current implementation, only scalar line-integrated quantities are
  supported.

- `itp_strategy :: Type{ITPSTRATEGY}`
  Interpolation strategy type. Only `itpScatter` is supported.

# Returns
- `GridBundle`
  A bundle containing:
  - `grids` : output `LineSamples` containers storing the interpolated scalar values
  - `names` : ordered quantity names matching the output grid order

# Throws
- `ArgumentError`: If `itp_strategy !== itpScatter`.
"""
function Partia.LineSamples_interpolation( :: MetalComputeBackend, grid_template :: LineSamples{3, TF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    itp_strategy === itpScatter || throw(ArgumentError(
        "LineSamples_interpolation only supports itpScatter. " *
        "Line-integrated samples do not have a well-defined query smoothing length ha, " *
        "so itpGather is not supported."
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
    LineSamples_interpolation(backend :: MetalComputeBackend, grid_template :: LineSamples{3, TF},
                              input :: AbstractInterpolationInput{3, TF}, LBVH :: LinearBVH{3, TF},
                              catalog :: InterpolationCatalog{3, N, 0, 0, 0, N},
                              itp_strategy :: Type{ITPSTRATEGY} = itpScatter)

Perform Metal-based SPH interpolation for an unstructured collection of line
samples using an externally supplied `LinearBVH`.

This routine assumes that `input` has already been reordered into the same
Morton leaf order used to build `LBVH`. Before interpolation begins, it checks
that the reordered spatial layout of `input` matches the leaf ordering stored in
`LBVH`. It then allocates output grids, builds the concise interpolation
catalog, copies the required data to Metal device memory, evaluates each line
sample on the GPU, and copies the interpolated grids back to host memory.

Each sample is interpreted as a line primitive defined by the corresponding
origin and direction stored in `grid_template`. For the `i`-th sample, the
interpolation kernel evaluates line-integrated quantities associated with that
line and stores the resulting scalar values into the output grids.

At present, this routine only supports `itpScatter`. For line-integrated
samples there is no well-defined query smoothing length `ha`, so
`itpGather` is rejected explicitly.

# Parameters
- `backend :: MetalComputeBackend`
  Execution backend specifying Metal-based interpolation.

- `grid_template :: LineSamples{3, TF}`
  Template sample container defining the dimensionality, line geometry, and
  output container layout.

- `input :: AbstractInterpolationInput{3, TF}`
  An interpolation input object containing particle positions, smoothing
  lengths, field values, and the SPH kernel. Its current ordering must already
  match the leaf ordering stored in `LBVH`.

- `LBVH :: LinearBVH{3, TF}`
  A prebuilt `LinearBVH` used for neighbour traversal during interpolation.

- `catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}`
  Interpolation catalog describing the requested output quantities.
  In the current implementation, only scalar line-integrated quantities are
  supported.

- `itp_strategy :: Type{ITPSTRATEGY}`
  Interpolation strategy type. Only `itpScatter` is supported.

# Returns
- `GridBundle`
  A bundle containing:
  - `grids` : output `LineSamples` containers storing the interpolated scalar values
  - `names` : ordered quantity names matching the output grid order

# Throws
- `ArgumentError`: If `itp_strategy !== itpScatter`, or if the leaf order stored
  in `LBVH` does not match the current spatial ordering of `input`.
"""
function Partia.LineSamples_interpolation( :: MetalComputeBackend, grid_template :: LineSamples{3, TF}, input :: INPUT, LBVH :: LinearBVH{3, TF}, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    itp_strategy === itpScatter || throw(ArgumentError(
        "LineSamples_interpolation only supports itpScatter. " *
        "Line-integrated samples do not have a well-defined query smoothing length ha, " *
        "so itpGather is not supported."
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

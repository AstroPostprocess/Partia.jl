######################################################################################

#     LineSamples interpolation wrappers

######################################################################################
"""
    LineSamples_interpolation!(
        grids :: NTuple{N, LS},
        input :: AbstractInterpolationInput{3, TF, VF},
        catalog :: InterpolationCatalog{3, N, 0, 0, 0, N},
        LBVH :: LinearBVH{3, TF, VF},
    )

Evaluate line-sample interpolation in place using a full catalog and a prebuilt
`LBVH`.

This wrapper is storage-agnostic. It prepares the concise catalog and dispatches
to the storage-specific core driver, such as the CPU `Vector` method in the main
package or GPU launch methods provided by extensions.

# Parameters
- `grids`: Preallocated line-sample output grids.
- `input`: Particle-side interpolation input.
- `catalog`: Scalar line-integrated quantities and output order.
- `LBVH`: Prebuilt hierarchy matching the current input order.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function LineSamples_interpolation!(grids :: NTuple{N, LS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, LBVH :: LinearBVH{3, TF, VF}) where {N, TF <: AbstractFloat, VF <: AbstractVector{TF}, LS <: LineSamples{3, TF, VF}, INPUT <: AbstractInterpolationInput{3, TF, VF}}
    # Get the name of columns
    names = catalog.ordered_names

    # Exit if nothing to do
    N == 0 && return GridBundle(grids, names)

    # Consistency test for LBVH when the storage supports cheap host checks
    _validate_interpolation_lbvh_leaf_order(input, LBVH)

    # Concise catalog
    catalog_consice = to_concise_catalog(catalog)

    LineSamples_interpolation_prepared!(grids, input, catalog_consice, LBVH)

    return GridBundle(grids, names)
end

"""
    LineSamples_interpolation(
        grid_template :: LineSamples{3, TF, VF},
        input :: AbstractInterpolationInput{3, TF, VF},
        catalog :: InterpolationCatalog{3, N, 0, 0, 0, N},
        LBVH :: LinearBVH{3, TF, VF},
    )

Allocate output line-sample grids and evaluate interpolation using a prebuilt
`LBVH`.

# Parameters
- `grid_template`: Geometry and storage template for every output grid.
- `input`: Particle-side interpolation input.
- `catalog`: Scalar line-integrated quantities and output order.
- `LBVH`: Prebuilt hierarchy matching the current input order.

# Returns
- `GridBundle`: Newly allocated line-sample grids in catalog order.
"""
function LineSamples_interpolation(grid_template :: LineSamples{3, TF, VF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, LBVH :: LinearBVH{3, TF, VF}) where {N, TF <: AbstractFloat, VF <: AbstractVector{TF}, INPUT <: AbstractInterpolationInput{3, TF, VF}}
    # Allocate output grids
    grids = ntuple(_ -> similar(grid_template), Val(N))

    return LineSamples_interpolation!(grids, input, catalog, LBVH)
end

######################################################################################

#     Wrapper that builds the LBVH

######################################################################################
"""
    LineSamples_interpolation!(
        grids :: NTuple{N, LS},
        input :: AbstractInterpolationInput{3, TF, VF},
        catalog :: InterpolationCatalog{3, N, 0, 0, 0, N},
    )

Build an `LBVH` from `input`, then evaluate line-sample interpolation in place.

# Parameters
- `grids`: Preallocated line-sample output grids.
- `input`: Particle-side interpolation input.
- `catalog`: Scalar line-integrated quantities and output order.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function LineSamples_interpolation!(grids :: NTuple{N, LS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}) where {N, TF <: AbstractFloat, VF <: AbstractVector{TF}, LS <: LineSamples{3, TF, VF}, INPUT <: AbstractInterpolationInput{3, TF, VF}}
    # Exit if nothing to do
    N == 0 && return GridBundle(grids, catalog.ordered_names)

    # Generate a Linear BVH structure for neighborhood searching
    LBVH = LinearBVH!(input, CodeType = UInt64)

    return LineSamples_interpolation!(grids, input, catalog, LBVH)
end

"""
    LineSamples_interpolation(
        grid_template :: LineSamples{3, TF, VF},
        input :: AbstractInterpolationInput{3, TF, VF},
        catalog :: InterpolationCatalog{3, N, 0, 0, 0, N},
    )

Allocate output line-sample grids, build an `LBVH` from `input`, and evaluate
interpolation.

# Parameters
- `grid_template`: Geometry and storage template for every output grid.
- `input`: Particle-side interpolation input.
- `catalog`: Scalar line-integrated quantities and output order.

# Returns
- `GridBundle`: Newly allocated line-sample grids in catalog order.
"""
function LineSamples_interpolation(grid_template :: LineSamples{3, TF, VF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}) where {N, TF <: AbstractFloat, VF <: AbstractVector{TF}, INPUT <: AbstractInterpolationInput{3, TF, VF}}
    # Allocate output grids
    grids = ntuple(_ -> similar(grid_template), Val(N))

    return LineSamples_interpolation!(grids, input, catalog)
end

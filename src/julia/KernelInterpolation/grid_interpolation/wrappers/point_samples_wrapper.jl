######################################################################################

#     PointSamples interpolation wrappers

######################################################################################
"""
    PointSamples_interpolation!(
        grids :: NTuple{L, PS},
        input :: AbstractInterpolationInput{3, TF, VF},
        catalog :: InterpolationCatalog{3, N, G, Div, C, L},
        LBVH :: LinearBVH{3, TF, VF},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Evaluate point-sample interpolation in place using a full catalog and a
prebuilt `LBVH`.

This wrapper is storage-agnostic. It prepares the concise catalog and dispatches
to the storage-specific core driver, such as the CPU `Vector` method in the main
package or GPU launch methods provided by extensions.

# Parameters
- `grids`: Preallocated point-sample output grids.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `LBVH`: Prebuilt hierarchy matching the current input order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, TF, VF}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, C, L, TF <: AbstractFloat, VF <: AbstractVector{TF}, PS <: PointSamples{3, TF, VF}, INPUT <: AbstractInterpolationInput{3, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Get the name of columns
    names = catalog.ordered_names

    # Exit if nothing to do
    L == 0 && return GridBundle(grids, names)

    # Consistency test for LBVH when the storage supports cheap host checks
    _validate_interpolation_lbvh_leaf_order(input, LBVH)

    # Concise catalog
    catalog_consice = to_concise_catalog(catalog)

    PointSamples_interpolation!(grids, input, catalog_consice, LBVH, itp_strategy)

    return GridBundle(grids, names)
end

"""
    PointSamples_interpolation!(
        grids :: NTuple{L, PS},
        input :: AbstractInterpolationInput{2, TF, VF},
        catalog :: InterpolationCatalog{2, N, G, Div, 0, L},
        LBVH :: LinearBVH{2, TF, VF},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Evaluate two-dimensional point-sample interpolation in place using a prebuilt
`LBVH`. The catalog must contain no curl requests.

# Parameters
- `grids`: Preallocated point-sample output grids.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `LBVH`: Prebuilt hierarchy matching the current input order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, TF, VF}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, L, TF <: AbstractFloat, VF <: AbstractVector{TF}, PS <: PointSamples{2, TF, VF}, INPUT <: AbstractInterpolationInput{2, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Get the name of columns
    names = catalog.ordered_names

    # Exit if nothing to do
    L == 0 && return GridBundle(grids, names)

    # Consistency test for LBVH when the storage supports cheap host checks
    _validate_interpolation_lbvh_leaf_order(input, LBVH)

    # Concise catalog
    catalog_consice = to_concise_catalog(catalog)

    PointSamples_interpolation!(grids, input, catalog_consice, LBVH, itp_strategy)

    return GridBundle(grids, names)
end

"""
    PointSamples_interpolation(
        grid_template :: PointSamples{2, TF, VF},
        input :: AbstractInterpolationInput{2, TF, VF},
        catalog :: InterpolationCatalog{2, N, G, Div, 0, L},
        LBVH :: LinearBVH{2, TF, VF},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Allocate two-dimensional output point-sample grids and evaluate interpolation
using a prebuilt `LBVH`. The catalog must contain no curl requests.

# Parameters
- `grid_template`: Geometry and storage template for every output grid.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `LBVH`: Prebuilt hierarchy matching the current input order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: Newly allocated point-sample grids in catalog order.
"""
function PointSamples_interpolation(grid_template :: PointSamples{2, TF, VF}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, TF, VF}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, L, TF <: AbstractFloat, VF <: AbstractVector{TF}, INPUT <: AbstractInterpolationInput{2, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Allocate output grids
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return PointSamples_interpolation!(grids, input, catalog, LBVH, itp_strategy)
end

"""
    PointSamples_interpolation(
        grid_template :: PointSamples{3, TF, VF},
        input :: AbstractInterpolationInput{3, TF, VF},
        catalog :: InterpolationCatalog{3, N, G, Div, C, L},
        LBVH :: LinearBVH{3, TF, VF},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Allocate output point-sample grids and evaluate interpolation using a prebuilt
`LBVH`.

# Parameters
- `grid_template`: Geometry and storage template for every output grid.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `LBVH`: Prebuilt hierarchy matching the current input order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: Newly allocated point-sample grids in catalog order.
"""
function PointSamples_interpolation(grid_template :: PointSamples{3, TF, VF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, TF, VF}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, C, L, TF <: AbstractFloat, VF <: AbstractVector{TF}, INPUT <: AbstractInterpolationInput{3, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Allocate output grids
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return PointSamples_interpolation!(grids, input, catalog, LBVH, itp_strategy)
end

######################################################################################

#     Wrapper that builds the LBVH

######################################################################################
"""
    PointSamples_interpolation!(
        grids :: NTuple{L, PS},
        input :: AbstractInterpolationInput{3, TF, VF},
        catalog :: InterpolationCatalog{3, N, G, Div, C, L},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Build an `LBVH` from `input`, then evaluate point-sample interpolation in
place.

# Parameters
- `grids`: Preallocated point-sample output grids.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, C, L, TF <: AbstractFloat, VF <: AbstractVector{TF}, PS <: PointSamples{3, TF, VF}, INPUT <: AbstractInterpolationInput{3, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Exit if nothing to do
    L == 0 && return GridBundle(grids, catalog.ordered_names)

    # Generate a Linear BVH structure for neighborhood searching
    LBVH = LinearBVH!(input, CodeType = UInt64)

    return PointSamples_interpolation!(grids, input, catalog, LBVH, itp_strategy)
end

"""
    PointSamples_interpolation!(
        grids :: NTuple{L, PS},
        input :: AbstractInterpolationInput{2, TF, VF},
        catalog :: InterpolationCatalog{2, N, G, Div, 0, L},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Build a two-dimensional `LBVH`, then evaluate point-sample interpolation in
place. The catalog must contain no curl requests.

# Parameters
- `grids`: Preallocated point-sample output grids.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, L, TF <: AbstractFloat, VF <: AbstractVector{TF}, PS <: PointSamples{2, TF, VF}, INPUT <: AbstractInterpolationInput{2, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Exit if nothing to do
    L == 0 && return GridBundle(grids, catalog.ordered_names)

    # Generate a Linear BVH structure for neighborhood searching
    LBVH = LinearBVH!(input, CodeType = UInt64)

    return PointSamples_interpolation!(grids, input, catalog, LBVH, itp_strategy)
end

"""
    PointSamples_interpolation(
        grid_template :: PointSamples{3, TF, VF},
        input :: AbstractInterpolationInput{3, TF, VF},
        catalog :: InterpolationCatalog{3, N, G, Div, C, L},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Allocate output point-sample grids, build an `LBVH` from `input`, and evaluate
interpolation.

# Parameters
- `grid_template`: Geometry and storage template for every output grid.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: Newly allocated point-sample grids in catalog order.
"""
function PointSamples_interpolation(grid_template :: PointSamples{3, TF, VF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, C, L, TF <: AbstractFloat, VF <: AbstractVector{TF}, INPUT <: AbstractInterpolationInput{3, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Allocate output grids
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return PointSamples_interpolation!(grids, input, catalog, itp_strategy)
end

"""
    PointSamples_interpolation(
        grid_template :: PointSamples{2, TF, VF},
        input :: AbstractInterpolationInput{2, TF, VF},
        catalog :: InterpolationCatalog{2, N, G, Div, 0, L},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Allocate two-dimensional output point-sample grids, build an `LBVH`, and
evaluate interpolation. The catalog must contain no curl requests.

# Parameters
- `grid_template`: Geometry and storage template for every output grid.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: Newly allocated point-sample grids in catalog order.
"""
function PointSamples_interpolation(grid_template :: PointSamples{2, TF, VF}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, L, TF <: AbstractFloat, VF <: AbstractVector{TF}, INPUT <: AbstractInterpolationInput{2, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Allocate output grids
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return PointSamples_interpolation!(grids, input, catalog, itp_strategy)
end

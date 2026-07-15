######################################################################################

#     StructuredGrid interpolation wrappers

######################################################################################
"""
    _flatten_structured_outputs(
        :: Type{COORD},
        grids :: NTuple{L, SG},
    ) where {L, COORD <: AbstractCoordinateSystem, SG <: StructuredGrid{3}}

Flatten structured output grids into point-sample views that share one
coordinate container.

The returned `PointSamples` reuse `vec(grids[i].grid)` as their value storage,
so interpolation writes directly back into the original structured grids.
"""
@inline function _flatten_structured_outputs( :: Type{COORD}, grids :: NTuple{L, SG}) where {L, COORD <: AbstractCoordinateSystem, SG <: StructuredGrid{3}}
    point_template = flatten(COORD, grids[1])
    return ntuple(i -> PointSamples(vec(grids[i].grid), point_template.coor), Val(L))
end

@inline function _flatten_structured_outputs( :: Type{COORD}, grids :: NTuple{L, SG}) where {L, COORD <: AbstractCoordinateSystem, SG <: StructuredGrid{2}}
    point_template = flatten(COORD, grids[1])
    return ntuple(i -> PointSamples(vec(grids[i].grid), point_template.coor), Val(L))
end

######################################################################################

#     Wrapper with a prebuilt LBVH

######################################################################################
"""
    StructuredGrid_interpolation!(
        :: Type{COORD},
        grids :: NTuple{L, SG},
        input :: AbstractInterpolationInput{2, TF, VF},
        catalog :: InterpolationCatalog{2, N, G, Div, 0, L},
        LBVH :: LinearBVH{2, TF, VF},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Evaluate two-dimensional structured-grid interpolation in place using a
prebuilt `LBVH`. The catalog must contain no curl requests.

# Parameters
- `COORD`: Coordinate-system tag used to flatten the structured grids.
- `grids`: Preallocated structured output grids.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `LBVH`: Prebuilt hierarchy matching the current input order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, TF, VF}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, VF <: AbstractVector{TF}, SG <: StructuredGrid{2, TF}, INPUT <: AbstractInterpolationInput{2, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Get the name of columns
    names = catalog.ordered_names

    # Exit if nothing to do
    L == 0 && return GridBundle(grids, names)

    # Consistency test for LBVH when the storage supports cheap host checks
    _validate_interpolation_lbvh_leaf_order(input, LBVH)

    # Make sure every grids share the same geometry
    if L > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output StructuredGrid grids must share the same coordinate axes. " *
            "Expected every grid to reuse the same axis vectors."
        ))
    end

    # Flatten into point-sample views sharing the same coordinate vectors
    point_grids = _flatten_structured_outputs(COORD, grids)

    # Interpolation
    catalog_consice = to_concise_catalog(catalog)
    PointSamples_interpolation!(point_grids, input, catalog_consice, LBVH, itp_strategy)

    return GridBundle(grids, names)
end

"""
    StructuredGrid_interpolation(
        :: Type{COORD},
        grid_template :: StructuredGrid{2, TF},
        input :: AbstractInterpolationInput{2, TF, VF},
        catalog :: InterpolationCatalog{2, N, G, Div, 0, L},
        LBVH :: LinearBVH{2, TF, VF},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Allocate two-dimensional structured output grids and evaluate interpolation
using a prebuilt `LBVH`. The catalog must contain no curl requests.

# Parameters
- `COORD`: Coordinate-system tag used to flatten the structured grid.
- `grid_template`: Geometry and storage template for every output grid.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `LBVH`: Prebuilt hierarchy matching the current input order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: Newly allocated structured grids in catalog order.
"""
function StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{2, TF}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, TF, VF}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, VF <: AbstractVector{TF}, INPUT <: AbstractInterpolationInput{2, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Allocate output grids
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return StructuredGrid_interpolation!(COORD, grids, input, catalog, LBVH, itp_strategy)
end

"""
    StructuredGrid_interpolation!(
        :: Type{COORD},
        grids :: NTuple{L, SG},
        input :: AbstractInterpolationInput{3, TF, VF},
        catalog :: InterpolationCatalog{3, N, G, Div, C, L},
        LBVH :: LinearBVH{3, TF, VF},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Evaluate structured-grid interpolation in place using a full catalog and a
prebuilt `LBVH`.

This wrapper flattens structured output grids into `PointSamples` views and
dispatches to the storage-specific point-sample core driver.

# Parameters
- `COORD`: Coordinate-system tag used to flatten the structured grids.
- `grids`: Preallocated structured output grids.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `LBVH`: Prebuilt hierarchy matching the current input order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.

"""
function StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, TF, VF}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, C, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, VF <: AbstractVector{TF}, SG <: StructuredGrid{3, TF}, INPUT <: AbstractInterpolationInput{3, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Get the name of columns
    names = catalog.ordered_names

    # Exit if nothing to do
    L == 0 && return GridBundle(grids, names)

    # Consistency test for LBVH when the storage supports cheap host checks
    _validate_interpolation_lbvh_leaf_order(input, LBVH)

    # Make sure every grids share the same geometry
    if L > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output StructuredGrid grids must share the same coordinate axes. " *
            "Expected every grid to reuse the same axis vectors."
        ))
    end

    # Flatten into point-sample views sharing the same coordinate vectors
    point_grids = _flatten_structured_outputs(COORD, grids)

    # Interpolation
    catalog_consice = to_concise_catalog(catalog)
    PointSamples_interpolation!(point_grids, input, catalog_consice, LBVH, itp_strategy)

    return GridBundle(grids, names)
end

"""
    StructuredGrid_interpolation(
        :: Type{COORD},
        grid_template :: StructuredGrid{3, TF},
        input :: AbstractInterpolationInput{3, TF, VF},
        catalog :: InterpolationCatalog{3, N, G, Div, C, L},
        LBVH :: LinearBVH{3, TF, VF},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Allocate output structured grids and evaluate interpolation using a prebuilt
`LBVH`.

# Parameters
- `COORD`: Coordinate-system tag used to flatten the structured grid.
- `grid_template`: Geometry and storage template for every output grid.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `LBVH`: Prebuilt hierarchy matching the current input order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: Newly allocated structured grids in catalog order.
"""
function StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{3, TF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, TF, VF}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, C, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, VF <: AbstractVector{TF}, INPUT <: AbstractInterpolationInput{3, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Allocate output grids
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return StructuredGrid_interpolation!(COORD, grids, input, catalog, LBVH, itp_strategy)
end

######################################################################################

#     Wrapper that builds the LBVH

######################################################################################
"""
    StructuredGrid_interpolation!(
        :: Type{COORD},
        grids :: NTuple{L, SG},
        input :: AbstractInterpolationInput{2, TF, VF},
        catalog :: InterpolationCatalog{2, N, G, Div, 0, L},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Build a two-dimensional `LBVH`, then evaluate structured-grid interpolation in
place. The catalog must contain no curl requests.

# Parameters
- `COORD`: Coordinate-system tag used to flatten the structured grids.
- `grids`: Preallocated structured output grids.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, VF <: AbstractVector{TF}, SG <: StructuredGrid{2, TF}, INPUT <: AbstractInterpolationInput{2, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Exit if nothing to do
    L == 0 && return GridBundle(grids, catalog.ordered_names)

    # Generate a Linear BVH structure for neighborhood searching
    LBVH = LinearBVH!(input, CodeType = UInt64)

    return StructuredGrid_interpolation!(COORD, grids, input, catalog, LBVH, itp_strategy)
end

"""
    StructuredGrid_interpolation(
        :: Type{COORD},
        grid_template :: StructuredGrid{2, TF},
        input :: AbstractInterpolationInput{2, TF, VF},
        catalog :: InterpolationCatalog{2, N, G, Div, 0, L},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Allocate two-dimensional structured output grids, build an `LBVH`, and
evaluate interpolation. The catalog must contain no curl requests.

# Parameters
- `COORD`: Coordinate-system tag used to flatten the structured grid.
- `grid_template`: Geometry and storage template for every output grid.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: Newly allocated structured grids in catalog order.
"""
function StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{2, TF}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, VF <: AbstractVector{TF}, INPUT <: AbstractInterpolationInput{2, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Allocate output grids
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return StructuredGrid_interpolation!(COORD, grids, input, catalog, itp_strategy)
end

"""
    StructuredGrid_interpolation!(
        :: Type{COORD},
        grids :: NTuple{L, SG},
        input :: AbstractInterpolationInput{3, TF, VF},
        catalog :: InterpolationCatalog{3, N, G, Div, C, L},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Build an `LBVH` from `input`, then evaluate structured-grid interpolation in
place.

# Parameters
- `COORD`: Coordinate-system tag used to flatten the structured grids.
- `grids`: Preallocated structured output grids.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, C, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, VF <: AbstractVector{TF}, SG <: StructuredGrid{3, TF}, INPUT <: AbstractInterpolationInput{3, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Exit if nothing to do
    L == 0 && return GridBundle(grids, catalog.ordered_names)

    # Generate a Linear BVH structure for neighborhood searching
    LBVH = LinearBVH!(input, CodeType = UInt64)

    return StructuredGrid_interpolation!(COORD, grids, input, catalog, LBVH, itp_strategy)
end

"""
    StructuredGrid_interpolation(
        :: Type{COORD},
        grid_template :: StructuredGrid{3, TF},
        input :: AbstractInterpolationInput{3, TF, VF},
        catalog :: InterpolationCatalog{3, N, G, Div, C, L},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Allocate output structured grids, build an `LBVH` from `input`, and evaluate
interpolation.

# Parameters
- `COORD`: Coordinate-system tag used to flatten the structured grid.
- `grid_template`: Geometry and storage template for every output grid.
- `input`: Particle-side interpolation input.
- `catalog`: Requested interpolation quantities and output order.
- `itp_strategy`: Gather or scatter interpolation strategy.

# Returns
- `GridBundle`: Newly allocated structured grids in catalog order.
"""
function StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{3, TF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, C, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, VF <: AbstractVector{TF}, INPUT <: AbstractInterpolationInput{3, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Allocate output grids
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return StructuredGrid_interpolation!(COORD, grids, input, catalog, itp_strategy)
end

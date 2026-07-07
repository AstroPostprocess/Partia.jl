######################################################################################

#     StructuredGrid interpolation wrappers

######################################################################################
"""
    _flatten_structured_outputs(
        :: Type{COORD},
        grids :: NTuple{L, SG},
    ) where {COORD <: AbstractCoordinateSystem, L, SG <: StructuredGrid{3}}

Flatten structured output grids into point-sample views that share one
coordinate container.

The returned `PointSamples` reuse `vec(grids[i].grid)` as their value storage,
so interpolation writes directly back into the original structured grids.
"""
@inline function _flatten_structured_outputs( :: Type{COORD}, grids :: NTuple{L, SG}) where {COORD <: AbstractCoordinateSystem, L, SG <: StructuredGrid{3}}
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
        input :: AbstractInterpolationInput{3, TF, VF},
        catalog :: InterpolationCatalog{3, N, G, Div, C, L},
        LBVH :: LinearBVH{3, TF, VF},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Evaluate structured-grid interpolation in place using a full catalog and a
prebuilt `LBVH`.

This wrapper flattens structured output grids into `PointSamples` views and
dispatches to the storage-specific point-sample core driver.
"""
function StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, TF, VF}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {COORD <: AbstractCoordinateSystem, N, G, Div, C, L, TF <: AbstractFloat, VF <: AbstractVector{TF}, SG <: StructuredGrid{3, TF}, INPUT <: AbstractInterpolationInput{3, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
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
"""
function StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{3, TF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, TF, VF}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {COORD <: AbstractCoordinateSystem, N, G, Div, C, L, TF <: AbstractFloat, VF <: AbstractVector{TF}, INPUT <: AbstractInterpolationInput{3, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
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
        input :: AbstractInterpolationInput{3, TF, VF},
        catalog :: InterpolationCatalog{3, N, G, Div, C, L},
        itp_strategy :: Type{ITPSTRATEGY} = itpScatter,
    )

Build an `LBVH` from `input`, then evaluate structured-grid interpolation in
place.
"""
function StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {COORD <: AbstractCoordinateSystem, N, G, Div, C, L, TF <: AbstractFloat, VF <: AbstractVector{TF}, SG <: StructuredGrid{3, TF}, INPUT <: AbstractInterpolationInput{3, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
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
"""
function StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{3, TF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {COORD <: AbstractCoordinateSystem, N, G, Div, C, L, TF <: AbstractFloat, VF <: AbstractVector{TF}, INPUT <: AbstractInterpolationInput{3, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Allocate output grids
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return StructuredGrid_interpolation!(COORD, grids, input, catalog, itp_strategy)
end

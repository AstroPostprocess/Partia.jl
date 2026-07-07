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

    PointSamples_interpolation_prepared!(grids, input, catalog_consice, LBVH, itp_strategy)

    return GridBundle(grids, names)
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
"""
function PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, C, L, TF <: AbstractFloat, VF <: AbstractVector{TF}, PS <: PointSamples{3, TF, VF}, INPUT <: AbstractInterpolationInput{3, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
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
"""
function PointSamples_interpolation(grid_template :: PointSamples{3, TF, VF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, G, Div, C, L, TF <: AbstractFloat, VF <: AbstractVector{TF}, INPUT <: AbstractInterpolationInput{3, TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Allocate output grids
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return PointSamples_interpolation!(grids, input, catalog, itp_strategy)
end

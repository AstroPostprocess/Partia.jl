######################################################################################

#     CUDA PointSamples interpolation wrappers

######################################################################################

"""
    PointSamples_interpolation!(grids, input, catalog, LBVH,
                                itp_strategy=itpScatter,
                                ::Val{ThreadsPerBlock}=Val(256))
    PointSamples_interpolation!(grids, input, catalog,
                                itp_strategy=itpScatter,
                                ::Val{ThreadsPerBlock}=Val(256))

Evaluate CUDA point-sample interpolation in place, optionally building the
`LBVH`. Both 2D and 3D CUDA inputs are supported.

# Parameters
- `grids`: Preallocated CUDA point-sample output grids.
- `input`: CUDA interpolation input.
- `catalog`: Requested quantities and output order.
- `LBVH`: Optional prebuilt hierarchy matching `input`.
- `itp_strategy`: Gather or scatter interpolation strategy.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function Partia.PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, L, TF <: AbstractFloat, PS <: PointSamples{2, TF, CuVector{TF}}, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Get the name of columns
    names = catalog.ordered_names

    # Exit if nothing to do
    L == 0 && return GridBundle(grids, names)

    # Consistency test for LBVH when the storage supports cheap host checks
    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)

    # Concise catalog
    catalog_consice = to_concise_catalog(catalog)
    Partia.PointSamples_interpolation!(grids, input, catalog_consice, LBVH, itp_strategy, Val(ThreadsPerBlock))

    return GridBundle(grids, names)
end

function Partia.PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, C, L, TF <: AbstractFloat, PS <: PointSamples{3, TF, CuVector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Get the name of columns
    names = catalog.ordered_names

    # Exit if nothing to do
    L == 0 && return GridBundle(grids, names)

    # Consistency test for LBVH when the storage supports cheap host checks
    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)

    # Concise catalog
    catalog_consice = to_concise_catalog(catalog)
    Partia.PointSamples_interpolation!(grids, input, catalog_consice, LBVH, itp_strategy, Val(ThreadsPerBlock))

    return GridBundle(grids, names)
end

function Partia.PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, L, TF <: AbstractFloat, PS <: PointSamples{2, TF, CuVector{TF}}, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Get the name of columns
    names = catalog.ordered_names

    # Exit if nothing to do
    L == 0 && return GridBundle(grids, names)

    # Build LBVH and reorder input into its leaf order
    LBVH = LinearBVH!(input, CodeType = UInt64)

    # Consistency test for LBVH when the storage supports cheap host checks
    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)

    # Concise catalog
    catalog_consice = to_concise_catalog(catalog)
    Partia.PointSamples_interpolation!(grids, input, catalog_consice, LBVH, itp_strategy, Val(ThreadsPerBlock))

    return GridBundle(grids, names)
end


function Partia.PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, C, L, TF <: AbstractFloat, PS <: PointSamples{3, TF, CuVector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Get the name of columns
    names = catalog.ordered_names

    # Exit if nothing to do
    L == 0 && return GridBundle(grids, names)

    # Build LBVH and reorder input into its leaf order
    LBVH = LinearBVH!(input, CodeType = UInt64)

    # Consistency test for LBVH when the storage supports cheap host checks
    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)

    # Concise catalog
    catalog_consice = to_concise_catalog(catalog)
    Partia.PointSamples_interpolation!(grids, input, catalog_consice, LBVH, itp_strategy, Val(ThreadsPerBlock))

    return GridBundle(grids, names)
end

"""
    PointSamples_interpolation(grid_template, input, catalog, [LBVH],
                               itp_strategy=itpScatter,
                               ::Val{ThreadsPerBlock}=Val(256))

Allocate CUDA point-sample outputs and evaluate 2D or 3D interpolation.

# Parameters
- `grid_template`: CUDA geometry and storage template.
- `input`: CUDA interpolation input.
- `catalog`: Requested quantities and output order.
- `LBVH`: Optional prebuilt hierarchy matching `input`.
- `itp_strategy`: Gather or scatter interpolation strategy.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `GridBundle`: Newly allocated CUDA point-sample grids.
"""
function Partia.PointSamples_interpolation(grid_template :: PointSamples{2, TF, CuVector{TF}}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, L, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.PointSamples_interpolation!(grids, input, catalog, LBVH, itp_strategy, Val(ThreadsPerBlock))
end


function Partia.PointSamples_interpolation(grid_template :: PointSamples{3, TF, CuVector{TF}}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, C, L, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.PointSamples_interpolation!(grids, input, catalog, LBVH, itp_strategy, Val(ThreadsPerBlock))
end


function Partia.PointSamples_interpolation(grid_template :: PointSamples{2, TF, CuVector{TF}}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, L, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.PointSamples_interpolation!(grids, input, catalog, itp_strategy, Val(ThreadsPerBlock))
end


function Partia.PointSamples_interpolation(grid_template :: PointSamples{3, TF, CuVector{TF}}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, C, L, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.PointSamples_interpolation!(grids, input, catalog, itp_strategy, Val(ThreadsPerBlock))
end

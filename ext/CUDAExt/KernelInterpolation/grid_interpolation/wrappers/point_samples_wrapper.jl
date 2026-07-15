######################################################################################

#     CUDA PointSamples interpolation wrappers

######################################################################################

function _point_samples_interpolation_cuda!(grids, input, catalog, LBVH, itp_strategy, threads_per_block)
    names = catalog.ordered_names
    length(names) == 0 && return GridBundle(grids, names)

    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)

    catalog_consice = to_concise_catalog(catalog)
    Partia.PointSamples_interpolation_prepared!(grids, input, catalog_consice, LBVH, itp_strategy, threads_per_block)

    return GridBundle(grids, names)
end

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
function Partia.PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, L, ThreadsPerBlock, TF <: AbstractFloat, PS <: PointSamples{2, TF, CuVector{TF}}, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    return _point_samples_interpolation_cuda!(grids, input, catalog, LBVH, itp_strategy, threads_per_block)
end

function Partia.PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, C, L, ThreadsPerBlock, TF <: AbstractFloat, PS <: PointSamples{3, TF, CuVector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    return _point_samples_interpolation_cuda!(grids, input, catalog, LBVH, itp_strategy, threads_per_block)
end

function Partia.PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, L, ThreadsPerBlock, TF <: AbstractFloat, PS <: PointSamples{2, TF, CuVector{TF}}, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    L == 0 && return GridBundle(grids, catalog.ordered_names)

    LBVH = LinearBVH!(input, CodeType = UInt64)

    return _point_samples_interpolation_cuda!(grids, input, catalog, LBVH, itp_strategy, threads_per_block)
end


function Partia.PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, C, L, ThreadsPerBlock, TF <: AbstractFloat, PS <: PointSamples{3, TF, CuVector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    L == 0 && return GridBundle(grids, catalog.ordered_names)

    LBVH = LinearBVH!(input, CodeType = UInt64)

    return _point_samples_interpolation_cuda!(grids, input, catalog, LBVH, itp_strategy, threads_per_block)
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
function Partia.PointSamples_interpolation(grid_template :: PointSamples{2, TF, CuVector{TF}}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, L, ThreadsPerBlock, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.PointSamples_interpolation!(grids, input, catalog, LBVH, itp_strategy, threads_per_block)
end


function Partia.PointSamples_interpolation(grid_template :: PointSamples{3, TF, CuVector{TF}}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, C, L, ThreadsPerBlock, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.PointSamples_interpolation!(grids, input, catalog, LBVH, itp_strategy, threads_per_block)
end


function Partia.PointSamples_interpolation(grid_template :: PointSamples{2, TF, CuVector{TF}}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, L, ThreadsPerBlock, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.PointSamples_interpolation!(grids, input, catalog, itp_strategy, threads_per_block)
end


function Partia.PointSamples_interpolation(grid_template :: PointSamples{3, TF, CuVector{TF}}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, C, L, ThreadsPerBlock, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.PointSamples_interpolation!(grids, input, catalog, itp_strategy, threads_per_block)
end

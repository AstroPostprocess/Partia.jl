######################################################################################

#     Metal PointSamples interpolation wrappers

######################################################################################

function _point_samples_interpolation_metal!(grids, input, catalog, LBVH, itp_strategy, threads_per_group)
    names = catalog.ordered_names
    length(names) == 0 && return GridBundle(grids, names)

    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)

    catalog_consice = to_concise_catalog(catalog)
    Partia.PointSamples_interpolation_prepared!(grids, input, catalog_consice, LBVH, itp_strategy, threads_per_group)

    return GridBundle(grids, names)
end

"""
    PointSamples_interpolation!(grids, input, catalog, [LBVH],
                                itp_strategy=itpScatter,
                                ::Val{ThreadsPerGroup}=Val(256))

Evaluate Metal point-sample interpolation in place, optionally building the
`LBVH`. Both 2D and 3D Metal inputs are supported.

# Parameters
- `grids`: Preallocated Metal point-sample output grids.
- `input`: Metal interpolation input.
- `catalog`: Requested quantities and output order.
- `LBVH`: Optional prebuilt hierarchy matching `input`.
- `itp_strategy`: Gather or scatter interpolation strategy.
- `::Val{ThreadsPerGroup}`: Metal threads per threadgroup.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function Partia.PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, Float32, MtlVector{Float32}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, L, ThreadsPerGroup, PS <: PointSamples{2, Float32, MtlVector{Float32}}, INPUT <: AbstractInterpolationInput{2, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    return _point_samples_interpolation_metal!(grids, input, catalog, LBVH, itp_strategy, threads_per_group)
end

function Partia.PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, Float32, MtlVector{Float32}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, C, L, ThreadsPerGroup, PS <: PointSamples{3, Float32, MtlVector{Float32}}, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    return _point_samples_interpolation_metal!(grids, input, catalog, LBVH, itp_strategy, threads_per_group)
end

function Partia.PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, L, ThreadsPerGroup, PS <: PointSamples{2, Float32, MtlVector{Float32}}, INPUT <: AbstractInterpolationInput{2, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    L == 0 && return GridBundle(grids, catalog.ordered_names)

    LBVH = LinearBVH!(input, CodeType = UInt64)

    return _point_samples_interpolation_metal!(grids, input, catalog, LBVH, itp_strategy, threads_per_group)
end


function Partia.PointSamples_interpolation!(grids :: NTuple{L, PS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, C, L, ThreadsPerGroup, PS <: PointSamples{3, Float32, MtlVector{Float32}}, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    L == 0 && return GridBundle(grids, catalog.ordered_names)

    LBVH = LinearBVH!(input, CodeType = UInt64)

    return _point_samples_interpolation_metal!(grids, input, catalog, LBVH, itp_strategy, threads_per_group)
end


"""
    PointSamples_interpolation(grid_template, input, catalog, [LBVH],
                               itp_strategy=itpScatter,
                               ::Val{ThreadsPerGroup}=Val(256))

Allocate Metal point-sample outputs and evaluate 2D or 3D interpolation.

# Parameters
- `grid_template`: Metal geometry and storage template.
- `input`: Metal interpolation input.
- `catalog`: Requested quantities and output order.
- `LBVH`: Optional prebuilt hierarchy matching `input`.
- `itp_strategy`: Gather or scatter interpolation strategy.
- `::Val{ThreadsPerGroup}`: Metal threads per threadgroup.

# Returns
- `GridBundle`: Newly allocated Metal point-sample grids.
"""
function Partia.PointSamples_interpolation(grid_template :: PointSamples{2, Float32, MtlVector{Float32}}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, Float32, MtlVector{Float32}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, L, ThreadsPerGroup, INPUT <: AbstractInterpolationInput{2, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.PointSamples_interpolation!(grids, input, catalog, LBVH, itp_strategy, threads_per_group)
end


function Partia.PointSamples_interpolation(grid_template :: PointSamples{3, Float32, MtlVector{Float32}}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, Float32, MtlVector{Float32}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, C, L, ThreadsPerGroup, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.PointSamples_interpolation!(grids, input, catalog, LBVH, itp_strategy, threads_per_group)
end


function Partia.PointSamples_interpolation(grid_template :: PointSamples{2, Float32, MtlVector{Float32}}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, L, ThreadsPerGroup, INPUT <: AbstractInterpolationInput{2, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.PointSamples_interpolation!(grids, input, catalog, itp_strategy, threads_per_group)
end


function Partia.PointSamples_interpolation(grid_template :: PointSamples{3, Float32, MtlVector{Float32}}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, C, L, ThreadsPerGroup, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.PointSamples_interpolation!(grids, input, catalog, itp_strategy, threads_per_group)
end

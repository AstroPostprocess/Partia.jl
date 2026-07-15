######################################################################################

#     Metal LineSamples interpolation wrappers

######################################################################################

function _line_samples_interpolation_metal!(grids, input, catalog, LBVH, threads_per_group)
    names = catalog.ordered_names
    length(names) == 0 && return GridBundle(grids, names)

    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)

    catalog_consice = to_concise_catalog(catalog)
    Partia.LineSamples_interpolation_prepared!(grids, input, catalog_consice, LBVH, threads_per_group)

    return GridBundle(grids, names)
end

"""
    LineSamples_interpolation!(grids, input, catalog, [LBVH],
                               ::Val{ThreadsPerGroup}=Val(256))

Evaluate Metal line-sample interpolation in place.

# Parameters
- `grids`: Preallocated Metal line-sample output grids.
- `input`: Metal interpolation input.
- `catalog`: Scalar line-integrated quantities and output order.
- `LBVH`: Optional prebuilt hierarchy matching `input`.
- `::Val{ThreadsPerGroup}`: Metal threads per threadgroup.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function Partia.LineSamples_interpolation!(grids :: NTuple{N, LS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, LBVH :: LinearBVH{3, Float32, MtlVector{Float32}}, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, ThreadsPerGroup, LS <: LineSamples{3, Float32, MtlVector{Float32}}, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}}
    return _line_samples_interpolation_metal!(grids, input, catalog, LBVH, threads_per_group)
end


function Partia.LineSamples_interpolation!(grids :: NTuple{N, LS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, ThreadsPerGroup, LS <: LineSamples{3, Float32, MtlVector{Float32}}, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}}
    N == 0 && return GridBundle(grids, catalog.ordered_names)

    LBVH = LinearBVH!(input, CodeType = UInt64)

    return _line_samples_interpolation_metal!(grids, input, catalog, LBVH, threads_per_group)
end


"""
    LineSamples_interpolation(grid_template, input, catalog, [LBVH],
                              ::Val{ThreadsPerGroup}=Val(256))

Allocate Metal line-sample outputs and evaluate interpolation.

# Parameters
- `grid_template`: Metal geometry and storage template.
- `input`: Metal interpolation input.
- `catalog`: Scalar line-integrated quantities and output order.
- `LBVH`: Optional prebuilt hierarchy matching `input`.
- `::Val{ThreadsPerGroup}`: Metal threads per threadgroup.

# Returns
- `GridBundle`: Newly allocated Metal line-sample grids.
"""
function Partia.LineSamples_interpolation(grid_template :: LineSamples{3, Float32, MtlVector{Float32}}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, LBVH :: LinearBVH{3, Float32, MtlVector{Float32}}, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, ThreadsPerGroup, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}}
    grids = ntuple(_ -> similar(grid_template), Val(N))

    return Partia.LineSamples_interpolation!(grids, input, catalog, LBVH, threads_per_group)
end


function Partia.LineSamples_interpolation(grid_template :: LineSamples{3, Float32, MtlVector{Float32}}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, ThreadsPerGroup, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}}
    grids = ntuple(_ -> similar(grid_template), Val(N))

    return Partia.LineSamples_interpolation!(grids, input, catalog, threads_per_group)
end

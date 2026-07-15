######################################################################################

#     Metal LineSamples interpolation wrappers

######################################################################################

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
function Partia.LineSamples_interpolation!(grids :: NTuple{N, LS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, LBVH :: LinearBVH{3, Float32, MtlVector{Float32}}, :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup, N, LS <: LineSamples{3, Float32, MtlVector{Float32}}, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}}
    names = catalog.ordered_names
    N == 0 && return GridBundle(grids, names)

    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)

    catalog_consice = to_concise_catalog(catalog)
    Partia.LineSamples_interpolation!(grids, input, catalog_consice, LBVH, Val(ThreadsPerGroup))

    return GridBundle(grids, names)
end


function Partia.LineSamples_interpolation!(grids :: NTuple{N, LS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup, N, LS <: LineSamples{3, Float32, MtlVector{Float32}}, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}}
    names = catalog.ordered_names
    N == 0 && return GridBundle(grids, names)

    LBVH = LinearBVH!(input, CodeType = UInt64)

    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)

    catalog_consice = to_concise_catalog(catalog)
    Partia.LineSamples_interpolation!(grids, input, catalog_consice, LBVH, Val(ThreadsPerGroup))

    return GridBundle(grids, names)
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
function Partia.LineSamples_interpolation(grid_template :: LineSamples{3, Float32, MtlVector{Float32}}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, LBVH :: LinearBVH{3, Float32, MtlVector{Float32}}, :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup, N, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}}
    grids = ntuple(_ -> similar(grid_template), Val(N))

    return Partia.LineSamples_interpolation!(grids, input, catalog, LBVH, Val(ThreadsPerGroup))
end


function Partia.LineSamples_interpolation(grid_template :: LineSamples{3, Float32, MtlVector{Float32}}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, :: Val{ThreadsPerGroup} = Val(256)) where {ThreadsPerGroup, N, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}}
    grids = ntuple(_ -> similar(grid_template), Val(N))

    return Partia.LineSamples_interpolation!(grids, input, catalog, Val(ThreadsPerGroup))
end

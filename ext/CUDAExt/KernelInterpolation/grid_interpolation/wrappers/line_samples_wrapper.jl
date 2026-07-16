######################################################################################

#     CUDA LineSamples interpolation wrappers

######################################################################################

"""
    LineSamples_interpolation!(grids, input, catalog, [LBVH],
                               ::Val{ThreadsPerBlock}=Val(256))

Evaluate CUDA line-sample interpolation in place.

# Parameters
- `grids`: Preallocated CUDA line-sample output grids.
- `input`: CUDA interpolation input.
- `catalog`: Scalar line-integrated quantities and output order.
- `LBVH`: Optional prebuilt hierarchy matching `input`.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function Partia.LineSamples_interpolation!(grids :: NTuple{N, LS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, TF <: AbstractFloat, LS <: LineSamples{3, TF, CuVector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}}
    names = catalog.ordered_names
    N == 0 && return GridBundle(grids, names)

    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)

    catalog_consice = to_concise_catalog(catalog)
    Partia.LineSamples_interpolation!(grids, input, catalog_consice, LBVH, Val(ThreadsPerBlock))

    return GridBundle(grids, names)
end


function Partia.LineSamples_interpolation!(grids :: NTuple{N, LS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, TF <: AbstractFloat, LS <: LineSamples{3, TF, CuVector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}}
    names = catalog.ordered_names
    N == 0 && return GridBundle(grids, names)

    LBVH = LinearBVH!(input, CodeType = UInt64)

    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)

    catalog_consice = to_concise_catalog(catalog)
    Partia.LineSamples_interpolation!(grids, input, catalog_consice, LBVH, Val(ThreadsPerBlock))

    return GridBundle(grids, names)
end


"""
    LineSamples_interpolation(grid_template, input, catalog, [LBVH],
                              ::Val{ThreadsPerBlock}=Val(256))

Allocate CUDA line-sample outputs and evaluate interpolation.

# Parameters
- `grid_template`: CUDA geometry and storage template.
- `input`: CUDA interpolation input.
- `catalog`: Scalar line-integrated quantities and output order.
- `LBVH`: Optional prebuilt hierarchy matching `input`.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `GridBundle`: Newly allocated CUDA line-sample grids.
"""
function Partia.LineSamples_interpolation(grid_template :: LineSamples{3, TF, CuVector{TF}}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}}
    grids = ntuple(_ -> similar(grid_template), Val(N))

    return Partia.LineSamples_interpolation!(grids, input, catalog, LBVH, Val(ThreadsPerBlock))
end


function Partia.LineSamples_interpolation(grid_template :: LineSamples{3, TF, CuVector{TF}}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}}
    grids = ntuple(_ -> similar(grid_template), Val(N))

    return Partia.LineSamples_interpolation!(grids, input, catalog, Val(ThreadsPerBlock))
end

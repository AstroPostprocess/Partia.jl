######################################################################################

#     CUDA LineSamples interpolation wrappers

######################################################################################

function _line_samples_interpolation_cuda!(grids, input, catalog, LBVH, threads_per_block)
    names = catalog.ordered_names
    length(names) == 0 && return GridBundle(grids, names)

    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)

    catalog_consice = to_concise_catalog(catalog)
    Partia.LineSamples_interpolation_prepared!(grids, input, catalog_consice, LBVH, threads_per_block)

    return GridBundle(grids, names)
end

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
function Partia.LineSamples_interpolation!(grids :: NTuple{N, LS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, ThreadsPerBlock, TF <: AbstractFloat, LS <: LineSamples{3, TF, CuVector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}}
    return _line_samples_interpolation_cuda!(grids, input, catalog, LBVH, threads_per_block)
end


function Partia.LineSamples_interpolation!(grids :: NTuple{N, LS}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, ThreadsPerBlock, TF <: AbstractFloat, LS <: LineSamples{3, TF, CuVector{TF}}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}}
    N == 0 && return GridBundle(grids, catalog.ordered_names)

    LBVH = LinearBVH!(input, CodeType = UInt64)

    return _line_samples_interpolation_cuda!(grids, input, catalog, LBVH, threads_per_block)
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
function Partia.LineSamples_interpolation(grid_template :: LineSamples{3, TF, CuVector{TF}}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, ThreadsPerBlock, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}}
    grids = ntuple(_ -> similar(grid_template), Val(N))

    return Partia.LineSamples_interpolation!(grids, input, catalog, LBVH, threads_per_block)
end


function Partia.LineSamples_interpolation(grid_template :: LineSamples{3, TF, CuVector{TF}}, input :: INPUT, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, ThreadsPerBlock, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}}
    grids = ntuple(_ -> similar(grid_template), Val(N))

    return Partia.LineSamples_interpolation!(grids, input, catalog, threads_per_block)
end

######################################################################################

#     CUDA StructuredGrid interpolation wrappers

######################################################################################

function _structured_grid_interpolation_cuda!( :: Type{COORD}, grids, input, catalog, LBVH, itp_strategy, threads_per_block) where {COORD <: AbstractCoordinateSystem}
    names = catalog.ordered_names
    length(names) == 0 && return GridBundle(grids, names)

    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)

    if length(grids) > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output StructuredGrid grids must share the same coordinate axes. " *
            "Expected every grid to reuse the same axis vectors."
        ))
    end

    point_grids = Partia.KernelInterpolation._flatten_structured_outputs(COORD, grids)

    catalog_consice = to_concise_catalog(catalog)
    Partia.PointSamples_interpolation_prepared!(point_grids, input, catalog_consice, LBVH, itp_strategy, threads_per_block)

    return GridBundle(grids, names)
end

"""
    StructuredGrid_interpolation!(COORD, grids, input, catalog, [LBVH],
                                  itp_strategy=itpScatter,
                                  ::Val{ThreadsPerBlock}=Val(256))

Evaluate CUDA structured-grid interpolation in place, optionally building the
`LBVH`. Both 2D and 3D CUDA inputs are supported.

# Parameters
- `COORD`: Coordinate-system tag used to flatten the structured grids.
- `grids`: Preallocated CUDA structured output grids.
- `input`: CUDA interpolation input.
- `catalog`: Requested quantities and output order.
- `LBVH`: Optional prebuilt hierarchy matching `input`.
- `itp_strategy`: Gather or scatter interpolation strategy.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function Partia.StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, L, ThreadsPerBlock, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, SG <: StructuredGrid{2, TF}, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    return _structured_grid_interpolation_cuda!(COORD, grids, input, catalog, LBVH, itp_strategy, threads_per_block)
end


function Partia.StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, C, L, ThreadsPerBlock, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, SG <: StructuredGrid{3, TF}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    return _structured_grid_interpolation_cuda!(COORD, grids, input, catalog, LBVH, itp_strategy, threads_per_block)
end


function Partia.StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, L, ThreadsPerBlock, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, SG <: StructuredGrid{2, TF}, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    L == 0 && return GridBundle(grids, catalog.ordered_names)

    LBVH = LinearBVH!(input, CodeType = UInt64)

    return _structured_grid_interpolation_cuda!(COORD, grids, input, catalog, LBVH, itp_strategy, threads_per_block)
end


function Partia.StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, C, L, ThreadsPerBlock, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, SG <: StructuredGrid{3, TF}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    L == 0 && return GridBundle(grids, catalog.ordered_names)

    LBVH = LinearBVH!(input, CodeType = UInt64)

    return _structured_grid_interpolation_cuda!(COORD, grids, input, catalog, LBVH, itp_strategy, threads_per_block)
end


"""
    StructuredGrid_interpolation(COORD, grid_template, input, catalog, [LBVH],
                                 itp_strategy=itpScatter,
                                 ::Val{ThreadsPerBlock}=Val(256))

Allocate CUDA structured-grid outputs and evaluate 2D or 3D interpolation.

# Parameters
- `COORD`: Coordinate-system tag used to flatten the structured grid.
- `grid_template`: CUDA geometry and storage template.
- `input`: CUDA interpolation input.
- `catalog`: Requested quantities and output order.
- `LBVH`: Optional prebuilt hierarchy matching `input`.
- `itp_strategy`: Gather or scatter interpolation strategy.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `GridBundle`: Newly allocated CUDA structured grids.
"""
function Partia.StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{2, TF}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, L, ThreadsPerBlock, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.StructuredGrid_interpolation!(COORD, grids, input, catalog, LBVH, itp_strategy, threads_per_block)
end


function Partia.StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{3, TF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, C, L, ThreadsPerBlock, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.StructuredGrid_interpolation!(COORD, grids, input, catalog, LBVH, itp_strategy, threads_per_block)
end


function Partia.StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{2, TF}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, L, ThreadsPerBlock, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.StructuredGrid_interpolation!(COORD, grids, input, catalog, itp_strategy, threads_per_block)
end


function Partia.StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{3, TF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_block :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, C, L, ThreadsPerBlock, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.StructuredGrid_interpolation!(COORD, grids, input, catalog, itp_strategy, threads_per_block)
end

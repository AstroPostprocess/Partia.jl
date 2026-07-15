######################################################################################

#     CUDA StructuredGrid interpolation wrappers

######################################################################################

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
function Partia.StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, SG <: StructuredGrid{2, TF}, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    names = catalog.ordered_names
    L == 0 && return GridBundle(grids, names)

    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)
    L > 1 && !same_coordinates(grids...) && throw(ArgumentError("All output StructuredGrid grids must share the same coordinate axes. Expected every grid to reuse the same axis vectors."))

    point_grids = Partia.KernelInterpolation._flatten_structured_outputs(COORD, grids)
    catalog_consice = to_concise_catalog(catalog)
    Partia.PointSamples_interpolation!(point_grids, input, catalog_consice, LBVH, itp_strategy, Val(ThreadsPerBlock))

    return GridBundle(grids, names)
end


function Partia.StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, C, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, SG <: StructuredGrid{3, TF}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    names = catalog.ordered_names
    L == 0 && return GridBundle(grids, names)

    Partia.KernelInterpolation._validate_interpolation_lbvh_leaf_order(input, LBVH)
    L > 1 && !same_coordinates(grids...) && throw(ArgumentError("All output StructuredGrid grids must share the same coordinate axes. Expected every grid to reuse the same axis vectors."))

    point_grids = Partia.KernelInterpolation._flatten_structured_outputs(COORD, grids)
    catalog_consice = to_concise_catalog(catalog)
    Partia.PointSamples_interpolation!(point_grids, input, catalog_consice, LBVH, itp_strategy, Val(ThreadsPerBlock))

    return GridBundle(grids, names)
end


function Partia.StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, SG <: StructuredGrid{2, TF}, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    L == 0 && return GridBundle(grids, catalog.ordered_names)

    LBVH = LinearBVH!(input, CodeType = UInt64)

    return Partia.StructuredGrid_interpolation!(COORD, grids, input, catalog, LBVH, itp_strategy, Val(ThreadsPerBlock))
end


function Partia.StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, C, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, SG <: StructuredGrid{3, TF}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    L == 0 && return GridBundle(grids, catalog.ordered_names)

    LBVH = LinearBVH!(input, CodeType = UInt64)

    return Partia.StructuredGrid_interpolation!(COORD, grids, input, catalog, LBVH, itp_strategy, Val(ThreadsPerBlock))
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
function Partia.StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{2, TF}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.StructuredGrid_interpolation!(COORD, grids, input, catalog, LBVH, itp_strategy, Val(ThreadsPerBlock))
end


function Partia.StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{3, TF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, C, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.StructuredGrid_interpolation!(COORD, grids, input, catalog, LBVH, itp_strategy, Val(ThreadsPerBlock))
end


function Partia.StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{2, TF}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.StructuredGrid_interpolation!(COORD, grids, input, catalog, itp_strategy, Val(ThreadsPerBlock))
end


function Partia.StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{3, TF}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, :: Val{ThreadsPerBlock} = Val(256)) where {ThreadsPerBlock, N, G, Div, C, L, COORD <: AbstractCoordinateSystem, TF <: AbstractFloat, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.StructuredGrid_interpolation!(COORD, grids, input, catalog, itp_strategy, Val(ThreadsPerBlock))
end

######################################################################################

#     Metal StructuredGrid interpolation wrappers

######################################################################################

function _structured_grid_interpolation_metal!( :: Type{COORD}, grids, input, catalog, LBVH, itp_strategy, threads_per_group) where {COORD <: AbstractCoordinateSystem}
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
    Partia.PointSamples_interpolation_prepared!(point_grids, input, catalog_consice, LBVH, itp_strategy, threads_per_group)

    return GridBundle(grids, names)
end

"""
    StructuredGrid_interpolation!(COORD, grids, input, catalog, [LBVH],
                                  itp_strategy=itpScatter,
                                  ::Val{ThreadsPerGroup}=Val(256))

Evaluate Metal structured-grid interpolation in place, optionally building the
`LBVH`. Both 2D and 3D Metal inputs are supported.

# Parameters
- `COORD`: Coordinate-system tag used to flatten the structured grids.
- `grids`: Preallocated Metal structured output grids.
- `input`: Metal interpolation input.
- `catalog`: Requested quantities and output order.
- `LBVH`: Optional prebuilt hierarchy matching `input`.
- `itp_strategy`: Gather or scatter interpolation strategy.
- `::Val{ThreadsPerGroup}`: Metal threads per threadgroup.

# Returns
- `GridBundle`: The supplied grids paired with the catalog output names.
"""
function Partia.StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, Float32, MtlVector{Float32}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, L, ThreadsPerGroup, COORD <: AbstractCoordinateSystem, SG <: StructuredGrid{2, Float32}, INPUT <: AbstractInterpolationInput{2, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    return _structured_grid_interpolation_metal!(COORD, grids, input, catalog, LBVH, itp_strategy, threads_per_group)
end


function Partia.StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, Float32, MtlVector{Float32}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, C, L, ThreadsPerGroup, COORD <: AbstractCoordinateSystem, SG <: StructuredGrid{3, Float32}, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    return _structured_grid_interpolation_metal!(COORD, grids, input, catalog, LBVH, itp_strategy, threads_per_group)
end


function Partia.StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, L, ThreadsPerGroup, COORD <: AbstractCoordinateSystem, SG <: StructuredGrid{2, Float32}, INPUT <: AbstractInterpolationInput{2, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    L == 0 && return GridBundle(grids, catalog.ordered_names)

    LBVH = LinearBVH!(input, CodeType = UInt64)

    return _structured_grid_interpolation_metal!(COORD, grids, input, catalog, LBVH, itp_strategy, threads_per_group)
end


function Partia.StructuredGrid_interpolation!( :: Type{COORD}, grids :: NTuple{L, SG}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, C, L, ThreadsPerGroup, COORD <: AbstractCoordinateSystem, SG <: StructuredGrid{3, Float32}, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    L == 0 && return GridBundle(grids, catalog.ordered_names)

    LBVH = LinearBVH!(input, CodeType = UInt64)

    return _structured_grid_interpolation_metal!(COORD, grids, input, catalog, LBVH, itp_strategy, threads_per_group)
end


"""
    StructuredGrid_interpolation(COORD, grid_template, input, catalog, [LBVH],
                                 itp_strategy=itpScatter,
                                 ::Val{ThreadsPerGroup}=Val(256))

Allocate Metal structured-grid outputs and evaluate 2D or 3D interpolation.

# Parameters
- `COORD`: Coordinate-system tag used to flatten the structured grid.
- `grid_template`: Metal geometry and storage template.
- `input`: Metal interpolation input.
- `catalog`: Requested quantities and output order.
- `LBVH`: Optional prebuilt hierarchy matching `input`.
- `itp_strategy`: Gather or scatter interpolation strategy.
- `::Val{ThreadsPerGroup}`: Metal threads per threadgroup.

# Returns
- `GridBundle`: Newly allocated Metal structured grids.
"""
function Partia.StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{2, Float32}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, LBVH :: LinearBVH{2, Float32, MtlVector{Float32}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, L, ThreadsPerGroup, COORD <: AbstractCoordinateSystem, INPUT <: AbstractInterpolationInput{2, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.StructuredGrid_interpolation!(COORD, grids, input, catalog, LBVH, itp_strategy, threads_per_group)
end


function Partia.StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{3, Float32}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, LBVH :: LinearBVH{3, Float32, MtlVector{Float32}}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, C, L, ThreadsPerGroup, COORD <: AbstractCoordinateSystem, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.StructuredGrid_interpolation!(COORD, grids, input, catalog, LBVH, itp_strategy, threads_per_group)
end


function Partia.StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{2, Float32}, input :: INPUT, catalog :: InterpolationCatalog{2, N, G, Div, 0, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, L, ThreadsPerGroup, COORD <: AbstractCoordinateSystem, INPUT <: AbstractInterpolationInput{2, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.StructuredGrid_interpolation!(COORD, grids, input, catalog, itp_strategy, threads_per_group)
end


function Partia.StructuredGrid_interpolation( :: Type{COORD}, grid_template :: StructuredGrid{3, Float32}, input :: INPUT, catalog :: InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter, threads_per_group :: Val{ThreadsPerGroup} = Val(256)) where {N, G, Div, C, L, ThreadsPerGroup, COORD <: AbstractCoordinateSystem, INPUT <: AbstractInterpolationInput{3, Float32, MtlVector{Float32}}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids = ntuple(_ -> similar(grid_template), Val(L))

    return Partia.StructuredGrid_interpolation!(COORD, grids, input, catalog, itp_strategy, threads_per_group)
end

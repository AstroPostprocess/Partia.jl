######################################################################################

#     CUDA PointSamples interpolation drivers

######################################################################################

#     Gather interpolation

######################################################################################
"""
    PointSamples_interpolation_prepared!(grids, input, catalog_consice, LBVH,
                                         itpGather,
                                         ::Val{ThreadsPerBlock}=Val(256))

Launch the prepared 2D CUDA gather-interpolation kernel.

# Parameters
- `grids`: Preallocated CUDA point-sample output grids.
- `input`: Prepared 2D CUDA interpolation input.
- `catalog_consice`: Concise execution catalog with no curl requests.
- `LBVH`: Prebuilt 2D hierarchy matching `input`.
- `itpGather`: Gather-strategy dispatch tag.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `nothing`: The supplied grids are updated in place.
"""
function Partia.PointSamples_interpolation_prepared!(grids :: NTuple{L, PS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{2, N, G, Div, 0}, LBVH :: LinearBVH{2, TF, CuVector{TF}}, :: Type{itpGather}, :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, L, ThreadsPerBlock, TF <: AbstractFloat, VF <: CuVector{TF}, VC <: NTuple{2, VF}, PS <: PointSamples{2, TF, VF, VC}, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}}
    L == 0 && return nothing

    if L > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output PointSamples grids must share the same point coordinates. " *
            "Expected every grid to reuse the same coordinate vectors."
        ))
    end

    npoints = length(grids[1])
    npoints == 0 && return nothing

    @cuda threads=(ThreadsPerBlock,) blocks=(cld(npoints, ThreadsPerBlock)) _point_samples_interpolation_kernel!(grids, input, catalog_consice, LBVH, itpGather)
    CUDA.synchronize()

    return nothing
end

"""
    PointSamples_interpolation_prepared!(grids, input, catalog_consice, LBVH,
                                         itpGather,
                                         ::Val{ThreadsPerBlock}=Val(256))

Launch the prepared 3D CUDA gather-interpolation kernel.

# Parameters
- `grids`: Preallocated CUDA point-sample output grids.
- `input`: Prepared 3D CUDA interpolation input.
- `catalog_consice`: Concise execution catalog.
- `LBVH`: Prebuilt 3D hierarchy matching `input`.
- `itpGather`: Gather-strategy dispatch tag.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `nothing`: The supplied grids are updated in place.
"""
function Partia.PointSamples_interpolation_prepared!(grids :: NTuple{L, PS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{3, N, G, Div, C}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, :: Type{itpGather}, :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, C, L, ThreadsPerBlock, TF <: AbstractFloat, VF <: CuVector{TF}, VC <: NTuple{3, VF}, PS <: PointSamples{3, TF, VF, VC}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}}
    L == 0 && return nothing

    if L > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output PointSamples grids must share the same point coordinates. " *
            "Expected every grid to reuse the same coordinate vectors."
        ))
    end

    npoints = length(grids[1])
    npoints == 0 && return nothing

    @cuda threads=(ThreadsPerBlock,) blocks=(cld(npoints, ThreadsPerBlock)) _point_samples_interpolation_kernel!(grids, input, catalog_consice, LBVH, itpGather)
    CUDA.synchronize()

    return nothing
end

######################################################################################

#     Scatter interpolation

######################################################################################
"""
    PointSamples_interpolation_prepared!(grids, input, catalog_consice, LBVH,
                                         itpScatter,
                                         ::Val{ThreadsPerBlock}=Val(256))

Launch the prepared 2D CUDA scatter-interpolation kernel.

# Parameters
- `grids`: Preallocated CUDA point-sample output grids.
- `input`: Prepared 2D CUDA interpolation input.
- `catalog_consice`: Concise execution catalog with no curl requests.
- `LBVH`: Prebuilt 2D hierarchy matching `input`.
- `itpScatter`: Scatter-strategy dispatch tag.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `nothing`: The supplied grids are updated in place.
"""
function Partia.PointSamples_interpolation_prepared!(grids :: NTuple{L, PS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{2, N, G, Div, 0}, LBVH :: LinearBVH{2, TF, CuVector{TF}}, :: Type{itpScatter}, :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, L, ThreadsPerBlock, TF <: AbstractFloat, VF <: CuVector{TF}, VC <: NTuple{2, VF}, PS <: PointSamples{2, TF, VF, VC}, INPUT <: AbstractInterpolationInput{2, TF, CuVector{TF}}}
    L == 0 && return nothing

    if L > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output PointSamples grids must share the same point coordinates. " *
            "Expected every grid to reuse the same coordinate vectors."
        ))
    end

    npoints = length(grids[1])
    npoints == 0 && return nothing

    @cuda threads=(ThreadsPerBlock,) blocks=(cld(npoints, ThreadsPerBlock)) _point_samples_interpolation_kernel!(grids, input, catalog_consice, LBVH, itpScatter)
    CUDA.synchronize()

    return nothing
end

"""
    PointSamples_interpolation_prepared!(grids, input, catalog_consice, LBVH,
                                         itpScatter,
                                         ::Val{ThreadsPerBlock}=Val(256))

Launch the prepared 3D CUDA scatter-interpolation kernel.

# Parameters
- `grids`: Preallocated CUDA point-sample output grids.
- `input`: Prepared 3D CUDA interpolation input.
- `catalog_consice`: Concise execution catalog.
- `LBVH`: Prebuilt 3D hierarchy matching `input`.
- `itpScatter`: Scatter-strategy dispatch tag.
- `::Val{ThreadsPerBlock}`: CUDA threads per block.

# Returns
- `nothing`: The supplied grids are updated in place.
"""
function Partia.PointSamples_interpolation_prepared!(grids :: NTuple{L, PS}, input :: INPUT, catalog_consice :: InterpolationCatalogConcise{3, N, G, Div, C}, LBVH :: LinearBVH{3, TF, CuVector{TF}}, :: Type{itpScatter}, :: Val{ThreadsPerBlock} = Val(256)) where {N, G, Div, C, L, ThreadsPerBlock, TF <: AbstractFloat, VF <: CuVector{TF}, VC <: NTuple{3, VF}, PS <: PointSamples{3, TF, VF, VC}, INPUT <: AbstractInterpolationInput{3, TF, CuVector{TF}}}
    L == 0 && return nothing

    if L > 1
        same_coordinates(grids...) || throw(ArgumentError(
            "All output PointSamples grids must share the same point coordinates. " *
            "Expected every grid to reuse the same coordinate vectors."
        ))
    end

    npoints = length(grids[1])
    npoints == 0 && return nothing

    @cuda threads=(ThreadsPerBlock,) blocks=(cld(npoints, ThreadsPerBlock)) _point_samples_interpolation_kernel!(grids, input, catalog_consice, LBVH, itpScatter)
    CUDA.synchronize()

    return nothing
end

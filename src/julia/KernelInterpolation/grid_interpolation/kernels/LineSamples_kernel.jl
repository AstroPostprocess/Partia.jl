# Interpolation kernels
## Line-integrated samples use particle-side smoothing lengths only
@inline function _line_samples_interpolation_kernel!(:: CPUComputeBackend, grids::NTuple{N, LineSamples{3, TF}}, i::Int, input::InterpolationInput{3, TF}, catalog_consice::InterpolationCatalogConcise{3, N, 0, 0, 0}, LBVH::LinearBVH, ::Type{itpScatter}) where {N, TF <: AbstractFloat}
    # Get line sample geometry
    @inbounds begin
        geometry = grids[1]

        xoa = geometry.origin[1][i]
        yoa = geometry.origin[2][i]
        zoa = geometry.origin[3][i]
        origin::NTuple{3, TF} = (xoa, yoa, zoa)

        xda = geometry.direction[1][i]
        yda = geometry.direction[2][i]
        zda = geometry.direction[3][i]
        direction::NTuple{3, TF} = (xda, yda, zda)
    end

    # Interpolation
    scalar_slots::NTuple{N, Int} = catalog_consice.scalar_slots
    scalar_snormalization::NTuple{N, Bool} = catalog_consice.scalar_snormalization
    scalars::NTuple{N, TF} = _line_integrated_quantities_interpolate_kernel(input, origin, direction, LBVH, scalar_slots, scalar_snormalization, itpScatter)

    # Store results
    if N > 0
        @inbounds for j in 1:N
            grids[j].grid[i] = scalars[j]
        end
    end

    return nothing
end
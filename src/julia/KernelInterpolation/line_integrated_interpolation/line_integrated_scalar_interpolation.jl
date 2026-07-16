@inline function _line_integrated_quantities_interpolate_kernel(input :: InterpolationInput{3, T}, origin :: NTuple{3, T}, direction :: NTuple{3, T}, LBVH :: LinearBVH, columns :: NTuple{M, Int}, ShepardNormalization :: NTuple{M, Bool}) where {T <: AbstractFloat, M}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    output :: MVector{M, T} = zero(MVector{M, T})
    S1 :: T = zero(T)

    leaf_idx :: Int = zero(Int)
    p2leaf_d2 :: T   = zero(T)
    hb :: T   = zero(T)

    LinearBoundingVolumeHierarchy.@LBVH_scatter_line_traversal LBVH origin direction Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]

            invhb = inv(hb)
            q_perp = Δr * invhb
            I = q_perp >= Kvalid ? zero(T) : invhb * invhb * lookup_line_integrated_kernel(Ktyp, q_perp)
            S1b = mb * I / ρb
            S1 += S1b

            @inbounds for j in 1:M
                column_idx = columns[j]
                Ab = input.quant[column_idx][leaf_idx]
                output[j] += Ab * S1b
            end
        end
        #########################################################
    end

    if iszero(S1)
        return ntuple(_ -> T(NaN32), Val(M))
    end

    invS1 = inv(S1)
    @inbounds for j in 1:M
        if ShepardNormalization[j]
            output[j] *= invS1
        end
    end

    return NTuple{M, T}(output)
end

@inline function _line_integrated_quantities_interpolate_kernel(input :: InterpolationInput{3, T}, origin :: NTuple{3, T}, direction :: NTuple{3, T}, LBVH :: LinearBVH) where {T <: AbstractFloat}
    val_len = Val(length(input.quant))
    columns = ntuple(identity, val_len)
    ShepardNormalization = ntuple(_ -> true, val_len)
    return _line_integrated_quantities_interpolate_kernel(input, origin, direction, LBVH, columns, ShepardNormalization)
end


@inline function _line_integrated_quantities_interpolate_kernel(input :: InterpolationSmoothingVolumeInput{3, T}, origin :: NTuple{3, T}, direction :: NTuple{3, T}, LBVH :: LinearBVH, columns :: NTuple{M, Int}, ShepardNormalization :: NTuple{M, Bool}) where {T <: AbstractFloat, M}
    hfact = input.hfact
    η = hfact * hfact * hfact
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # The smoothing-volume relation replaces m_b/rho_b with h_b^3/hfact^3.
    # lookup_line_integrated_kernel already includes the 3D normalization.
    # Since the full line-integrated kernel scales as h_b^-2, the per-neighbor
    # loop keeps h_b * ∫W(q_perp), while hfact^-3 is applied once after
    # accumulation for non-Shepard outputs.
    prefactor = inv(η)

    output :: MVector{M, T} = zero(MVector{M, T})
    S1 :: T = zero(T)

    leaf_idx :: Int = zero(Int)
    p2leaf_d2 :: T   = zero(T)
    hb :: T   = zero(T)

    LinearBoundingVolumeHierarchy.@LBVH_scatter_line_traversal LBVH origin direction Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            q_perp = Δr / hb

            ∫wb = q_perp >= Kvalid ? zero(T) : lookup_line_integrated_kernel(Ktyp, q_perp)

            S1b = hb * ∫wb
            S1 += S1b

            @inbounds for j in 1:M
                column_idx = columns[j]
                Ab = input.quant[column_idx][leaf_idx]
                output[j] += Ab * S1b
            end
        end
        #########################################################
    end

    if iszero(S1)
        return ntuple(_ -> T(NaN32), Val(M))
    end

    invS1 = inv(S1)
    @inbounds for j in 1:M
        if ShepardNormalization[j]
            output[j] *= invS1
        else
            output[j] *= prefactor
        end
    end

    return NTuple{M, T}(output)
end

@inline function _line_integrated_quantities_interpolate_kernel(input :: InterpolationSmoothingVolumeInput{3, T}, origin :: NTuple{3, T}, direction :: NTuple{3, T}, LBVH :: LinearBVH) where {T <: AbstractFloat}
    val_len = Val(length(input.quant))
    columns = ntuple(identity, val_len)
    ShepardNormalization = ntuple(_ -> true, val_len)
    return _line_integrated_quantities_interpolate_kernel(input, origin, direction, LBVH, columns, ShepardNormalization)
end

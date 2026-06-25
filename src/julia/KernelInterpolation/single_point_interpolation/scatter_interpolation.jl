
@inline function _general_quantity_interpolate_kernel(
                        input :: InterpolationInput{3, T, V, Ktyp, NCOLUMN},
                        reference_point :: NTuple{3,T},
                        LBVH :: LinearBVH,
                        catalog :: InterpolationCatalogConcise{3,N,G,D,C}) :: Tuple{NTuple{N,T}, NTuple{G,NTuple{3,T}}, NTuple{D,T}, NTuple{C,NTuple{3,T}}} where {N, G, D, C, T <: AbstractFloat, V <: AbstractVector{T}, Ktyp <: AbstractSPHKernel, NCOLUMN}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Kvalid = KernelFunctionValid(Ktyp, T)
    # Scatter uses each neighbour smoothing length `hb`; only Cnorm is common.
    Cnorm = KernelFunctionnorm(Ktyp, Val(3), T)
    ShepardNormalization = catalog.scalar_snormalization
    x = get_xcoord(input)
    y = get_ycoord(input)
    z = get_zcoord(input)
    @inbounds begin
        xa = reference_point[1]; ya = reference_point[2]; za = reference_point[3];
    end

    # Initialize counter
    ## Shepard Normalization
    S1 :: T = zero(T)

    ## Scalars
    scalars :: MVector{N, T} = zero(MVector{N, T})

    ## Gradients
    gradients_c :: MVector{G, SVector{3,T}} = MVector{G, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(G)))
    gradients_scalars :: MVector{G, T} = zero(MVector{G, T})                                                                  # Scalar that is used for estimating gradients

    ## Divergences
    divergences_c :: MVector{D, T} = zero(MVector{D, T})
    divergences_scalars :: MVector{D, SVector{3,T}} = MVector{D, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(D)))       # Scalars that is used for estimating divergnece

    ## Curls
    curls_c :: MVector{C, SVector{3,T}} = MVector{C, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(C)))
    curls_scalars :: MVector{C, SVector{3,T}} = MVector{C, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(C)))             # Scalars that is used for estimating curls

    ## Correction reduction
    ## Kernel-gradient term used by the Shepard-consistent derivative
    ## correction. Keep scalar components to avoid GPU-side StaticArray stores.
    correction_x :: T = zero(T)
    correction_y :: T = zero(T)
    correction_z :: T = zero(T)

    # Traversal
    leaf_idx :: Int = zero(Int)
    p2leaf_d2 :: T   = zero(T)
    hb :: T   = zero(T)

    LinearBoundingVolumeHierarchy.@LBVH_scatter_point_traversal LBVH reference_point Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            xb = x[leaf_idx]; yb = y[leaf_idx]; zb = z[leaf_idx]
            Δx = xa - xb
            Δy = ya - yb
            Δz = za - zb

            Δr = sqrt(p2leaf_d2)
            if iszero(Δr)
                Δx̂ = zero(T)
                Δŷ = zero(T)
                Δẑ = zero(T)
            else
                invΔr = inv(Δr)
                Δx̂ = Δx * invΔr
                Δŷ = Δy * invΔr
                Δẑ = Δz * invΔr
            end

            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]
            # General volume element for standard SPH interpolation.
            mblρb = mb / ρb

            # Scatter evaluates the full h-scaling per neighbour.
            invhb = inv(hb)
            invhb2 = invhb * invhb
            invhb3 = invhb2 * invhb
            invhb4 = invhb3 * invhb
            q = Δr * invhb

            wb = K(q)
            ∂wb = KernelFunctionDiff(Ktyp, q)
            ∂xwb = ∂wb * Δx̂
            ∂ywb = ∂wb * Δŷ
            ∂zwb = ∂wb * Δẑ

            # Full kernel weights:
            ## W  = Cnorm / hb^3 * w(q)
            ## dW = Cnorm / hb^4 * dw(q) * rhat
            mlρCnormlhb3 = mblρb * Cnorm * invhb3
            mlρCnormlhb4 = mblρb * Cnorm * invhb4

            mbWlρb = mlρCnormlhb3 * wb
            mb∂xWlρb = mlρCnormlhb4 * ∂xwb
            mb∂yWlρb = mlρCnormlhb4 * ∂ywb
            mb∂zWlρb = mlρCnormlhb4 * ∂zwb

            correction_x += mb∂xWlρb
            correction_y += mb∂yWlρb
            correction_z += mb∂zWlρb

            # Shepard Normalization
            S1 += mbWlρb

            # Scalar interpolations
            @inbounds for j in 1:N
                slot = catalog.scalar_slots[j]
                Ab = input.quant[slot][leaf_idx]
                scalars[j] += Ab * mbWlρb
            end

            # Gradient interpolations
            @inbounds for j in 1:G
                slot = catalog.grad_slots[j]
                Ab = input.quant[slot][leaf_idx]

                Abmb∂xWlρb = Ab * mb∂xWlρb
                Abmb∂yWlρb = Ab * mb∂yWlρb
                Abmb∂zWlρb = Ab * mb∂zWlρb

                gradients_c[j] += SVector{3,T}(Abmb∂xWlρb, Abmb∂yWlρb, Abmb∂zWlρb)
                gradients_scalars[j] += Ab * mbWlρb
            end

            # Divergence interpolations
            @inbounds for j in 1:D
                slot = catalog.div_slots[j]
                Ax_column_idx, Ay_column_idx, Az_column_idx = slot
                Axb = input.quant[Ax_column_idx][leaf_idx]
                Ayb = input.quant[Ay_column_idx][leaf_idx]
                Azb = input.quant[Az_column_idx][leaf_idx]

                divergences_c[j] += Axb * mb∂xWlρb + Ayb * mb∂yWlρb + Azb * mb∂zWlρb

                Axa = Axb * mbWlρb
                Aya = Ayb * mbWlρb
                Aza = Azb * mbWlρb
                divergences_scalars[j] += SVector{3,T}(Axa, Aya, Aza)
            end

            # Curl interpolations
            @inbounds for j in 1:C
                slot = catalog.curl_slots[j]
                Ax_column_idx, Ay_column_idx, Az_column_idx = slot
                Axb = input.quant[Ax_column_idx][leaf_idx]
                Ayb = input.quant[Ay_column_idx][leaf_idx]
                Azb = input.quant[Az_column_idx][leaf_idx]

                Axbmb∂yWlρb = Axb * mb∂yWlρb
                Axbmb∂zWlρb = Axb * mb∂zWlρb
                Aybmb∂xWlρb = Ayb * mb∂xWlρb
                Aybmb∂zWlρb = Ayb * mb∂zWlρb
                Azbmb∂xWlρb = Azb * mb∂xWlρb
                Azbmb∂yWlρb = Azb * mb∂yWlρb

                curls_c[j] += SVector{3,T}(Aybmb∂zWlρb - Azbmb∂yWlρb, Azbmb∂xWlρb - Axbmb∂zWlρb, Axbmb∂yWlρb - Aybmb∂xWlρb)

                Axa = Axb * mbWlρb
                Aya = Ayb * mbWlρb
                Aza = Azb * mbWlρb
                curls_scalars[j] += SVector{3,T}(Axa, Aya, Aza)
            end
        end
        #########################################################

    end

    # Preparing output
    if iszero(S1)
        output = (ntuple(_ -> T(NaN), Val(N)), ntuple(_ -> (T(NaN), T(NaN), T(NaN)), Val(G)), ntuple(i -> T(NaN), Val(D)), ntuple(i -> (T(NaN), T(NaN), T(NaN)), Val(C)))
        return output
    end

    # Shepard normalization
    invS1 = inv(S1)

    @inbounds for j in 1:N
        if ShepardNormalization[j]
            scalars[j] *= invS1
        end
    end

    # Initialize output containers
    gradients :: MVector{G, NTuple{3, T}} = MVector{G, NTuple{3, T}}(ntuple(_ -> (zero(T), zero(T), zero(T)), Val(G)))
    divergences :: MVector{D, T} = zero(MVector{D, T})
    curls :: MVector{C, NTuple{3, T}} = MVector{C, NTuple{3, T}}(ntuple(_ -> (zero(T), zero(T), zero(T)), Val(C)))

    # Construct gradients
    @inbounds for j in 1:G
        A    = gradients_scalars[j] * invS1

        # Final result
        ∇Ax = gradients_c[j][1] - A * correction_x
        ∇Ay = gradients_c[j][2] - A * correction_y
        ∇Az = gradients_c[j][3] - A * correction_z

        gradients[j] = (∇Ax, ∇Ay, ∇Az)
    end

    # Construct divergences
    @inbounds for j in 1:D
        Ax   = divergences_scalars[j][1] * invS1
        Ay   = divergences_scalars[j][2] * invS1
        Az   = divergences_scalars[j][3] * invS1

        # Final result
        ∇A = divergences_c[j] - Ax * correction_x - Ay * correction_y - Az * correction_z

        divergences[j] = ∇A
    end

    # Construct curls
    @inbounds for j in 1:C
        Ax   = curls_scalars[j][1] * invS1
        Ay   = curls_scalars[j][2] * invS1
        Az   = curls_scalars[j][3] * invS1

        m∂xWlρ = correction_x
        m∂yWlρ = correction_y
        m∂zWlρ = correction_z

        # Final result
        Axm∂yWlρ = Ax * m∂yWlρ
        Axm∂zWlρ = Ax * m∂zWlρ
        Aym∂xWlρ = Ay * m∂xWlρ
        Aym∂zWlρ = Ay * m∂zWlρ
        Azm∂xWlρ = Az * m∂xWlρ
        Azm∂yWlρ = Az * m∂yWlρ

        ∇Ax = Aym∂zWlρ - Azm∂yWlρ - curls_c[j][1]
        ∇Ay = Azm∂xWlρ - Axm∂zWlρ - curls_c[j][2]
        ∇Az = Axm∂yWlρ - Aym∂xWlρ - curls_c[j][3]

        curls[j] = (∇Ax, ∇Ay, ∇Az)
    end

    scalars_out = ntuple(i -> scalars[i], Val(N))
    gradients_out = ntuple(i -> gradients[i], Val(G))
    divergences_out = ntuple(i -> divergences[i], Val(D))
    curls_out = ntuple(i -> curls[i], Val(C))

    output = (scalars_out, gradients_out, divergences_out, curls_out)
    return output
end

@inline function _general_quantity_interpolate_kernel(
                        input :: InterpolationSmoothingVolumeInput{3, T, V, Ktyp, NCOLUMN},
                        reference_point :: NTuple{3,T},
                        LBVH :: LinearBVH,
                        catalog :: InterpolationCatalogConcise{3,N,G,D,C}) :: Tuple{NTuple{N,T}, NTuple{G,NTuple{3,T}}, NTuple{D,T}, NTuple{C,NTuple{3,T}}} where {N, G, D, C, T <: AbstractFloat, V <: AbstractVector{T}, Ktyp <: AbstractSPHKernel, NCOLUMN}
    # Prepare for interpolation
    hfact = input.hfact
    η = hfact * hfact * hfact
    K = input.smoothed_kernel
    Kvalid = KernelFunctionValid(Ktyp, T)
    ShepardNormalization = catalog.scalar_snormalization
    x = get_xcoord(input)
    y = get_ycoord(input)
    z = get_zcoord(input)
    @inbounds begin
        xa = reference_point[1]; ya = reference_point[2]; za = reference_point[3];
    end

    # Prefactor for all interpolation (hfact^-3 Cnorm)
    ## The smoothing-volume relation replaces m_b/rho_b with h_b^3/hfact^3.
    ## For scatter, h_b cancels the kernel h_b^-3 and leaves a common factor.
    invη = inv(η)
    prefactor = KernelFunctionnorm(Ktyp, Val(3), T) * invη


    # Initialize counter
    ## Shepard Normalization
    S1 :: T = zero(T)

    ## Scalars
    scalars :: MVector{N, T} = zero(MVector{N, T})

    ## Gradients
    gradients_c :: MVector{G, SVector{3,T}} = MVector{G, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(G)))
    gradients_scalars :: MVector{G, T} = zero(MVector{G, T})                                                                  # Scalar that is used for estimating gradients

    ## Divergences
    divergences_c :: MVector{D, T} = zero(MVector{D, T})
    divergences_scalars :: MVector{D, SVector{3,T}} = MVector{D, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(D)))       # Scalars that is used for estimating divergnece

    ## Curls
    curls_c :: MVector{C, SVector{3,T}} = MVector{C, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(C)))
    curls_scalars :: MVector{C, SVector{3,T}} = MVector{C, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(C)))             # Scalars that is used for estimating curls

    ## Correction reduction
    ## Same correction as the general path after cancelling the smoothing-volume
    ## factors; the common prefactor is applied when constructing outputs.
    correction_x :: T = zero(T)
    correction_y :: T = zero(T)
    correction_z :: T = zero(T)

    # Traversal
    leaf_idx :: Int = zero(Int)
    p2leaf_d2 :: T   = zero(T)
    hb :: T   = zero(T)

    LinearBoundingVolumeHierarchy.@LBVH_scatter_point_traversal LBVH reference_point Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            xb = x[leaf_idx]; yb = y[leaf_idx]; zb = z[leaf_idx]
            Δx = xa - xb
            Δy = ya - yb
            Δz = za - zb

            Δr = sqrt(p2leaf_d2)
            if iszero(Δr)
                Δx̂ = zero(T)
                Δŷ = zero(T)
                Δẑ = zero(T)
            else
                invΔr = inv(Δr)
                Δx̂ = Δx * invΔr
                Δŷ = Δy * invΔr
                Δẑ = Δz * invΔr
            end

            # Scatter uses the neighbour smoothing length directly.
            invhb = inv(hb)
            q  = Δr * invhb

            wb = K(q)
            ∂wb = KernelFunctionDiff(Ktyp, q)
            ∂xwb = ∂wb * Δx̂
            ∂ywb = ∂wb * Δŷ
            ∂zwb = ∂wb * Δẑ

            # Raw derivative shape after h_b cancellation:
            ## (h_b^3 / hfact^3) * (Cnorm / h_b^4) -> Cnorm/hfact^3 * inv(h_b)
            ∂xwblhb = invhb * ∂xwb
            ∂ywblhb = invhb * ∂ywb
            ∂zwblhb = invhb * ∂zwb

            correction_x += ∂xwblhb
            correction_y += ∂ywblhb
            correction_z += ∂zwblhb

            # Shepard Normalization
            S1 += wb

            # Scalar interpolations
            @inbounds for j in 1:N
                slot = catalog.scalar_slots[j]
                Ab = input.quant[slot][leaf_idx]
                scalars[j] += Ab * wb
            end

            # Gradient interpolations
            @inbounds for j in 1:G
                slot = catalog.grad_slots[j]
                Ab = input.quant[slot][leaf_idx]

                Ab∂xwblhb = Ab * ∂xwblhb
                Ab∂ywblhb = Ab * ∂ywblhb
                Ab∂zwblhb = Ab * ∂zwblhb

                gradients_c[j] += SVector{3,T}(Ab∂xwblhb, Ab∂ywblhb, Ab∂zwblhb)
                gradients_scalars[j] += Ab * wb
            end

            # Divergence interpolations
            @inbounds for j in 1:D
                slot = catalog.div_slots[j]
                Ax_column_idx, Ay_column_idx, Az_column_idx = slot
                Axb = input.quant[Ax_column_idx][leaf_idx]
                Ayb = input.quant[Ay_column_idx][leaf_idx]
                Azb = input.quant[Az_column_idx][leaf_idx]

                divergences_c[j] += Axb * ∂xwblhb + Ayb * ∂ywblhb + Azb * ∂zwblhb

                Axa = Axb * wb
                Aya = Ayb * wb
                Aza = Azb * wb
                divergences_scalars[j] += SVector{3,T}(Axa, Aya, Aza)
            end

            # Curl interpolations
            @inbounds for j in 1:C
                slot = catalog.curl_slots[j]
                Ax_column_idx, Ay_column_idx, Az_column_idx = slot
                Axb = input.quant[Ax_column_idx][leaf_idx]
                Ayb = input.quant[Ay_column_idx][leaf_idx]
                Azb = input.quant[Az_column_idx][leaf_idx]

                Axb∂ywblhb = Axb * ∂ywblhb
                Axb∂zwblhb = Axb * ∂zwblhb
                Ayb∂xwblhb = Ayb * ∂xwblhb
                Ayb∂zwblhb = Ayb * ∂zwblhb
                Azb∂xwblhb = Azb * ∂xwblhb
                Azb∂ywblhb = Azb * ∂ywblhb

                curls_c[j] += SVector{3,T}(Ayb∂zwblhb - Azb∂ywblhb, Azb∂xwblhb - Axb∂zwblhb, Axb∂ywblhb - Ayb∂xwblhb)

                Axa = Axb * wb
                Aya = Ayb * wb
                Aza = Azb * wb
                curls_scalars[j] += SVector{3,T}(Axa, Aya, Aza)
            end
        end
        #########################################################
    end

    # Preparing output
    if iszero(S1)
        output = (ntuple(_ -> T(NaN), Val(N)), ntuple(_ -> (T(NaN), T(NaN), T(NaN)), Val(G)), ntuple(i -> T(NaN), Val(D)), ntuple(i -> (T(NaN), T(NaN), T(NaN)), Val(C)))
        return output
    end

    # Shepard normalization
    invS1 = inv(S1)

    @inbounds for j in 1:N
        if ShepardNormalization[j]
            scalars[j] *= invS1
        else
            scalars[j] *= prefactor
        end
    end

    # Initialize output containers
    gradients :: MVector{G, NTuple{3, T}} = MVector{G, NTuple{3, T}}(ntuple(_ -> (zero(T), zero(T), zero(T)), Val(G)))
    divergences :: MVector{D, T} = zero(MVector{D, T})
    curls :: MVector{C, NTuple{3, T}} = MVector{C, NTuple{3, T}}(ntuple(_ -> (zero(T), zero(T), zero(T)), Val(C)))

    # Construct gradients
    @inbounds for j in 1:G
        A    = gradients_scalars[j] * invS1

        # Final result
        ∇Ax = prefactor * (gradients_c[j][1] - A * correction_x)
        ∇Ay = prefactor * (gradients_c[j][2] - A * correction_y)
        ∇Az = prefactor * (gradients_c[j][3] - A * correction_z)

        gradients[j] = (∇Ax, ∇Ay, ∇Az)
    end

    # Construct divergences
    @inbounds for j in 1:D
        Ax   = divergences_scalars[j][1] * invS1
        Ay   = divergences_scalars[j][2] * invS1
        Az   = divergences_scalars[j][3] * invS1

        # Final result
        ∇A = prefactor * (divergences_c[j] - Ax * correction_x - Ay * correction_y - Az * correction_z)
        divergences[j] = ∇A
    end

    # Construct curls
    @inbounds for j in 1:C
        Ax   = curls_scalars[j][1] * invS1
        Ay   = curls_scalars[j][2] * invS1
        Az   = curls_scalars[j][3] * invS1

        ∂xwlhb = correction_x
        ∂ywlhb = correction_y
        ∂zwlhb = correction_z

        # Final result
        Ax∂ywlhb = Ax * ∂ywlhb
        Ax∂zwlhb = Ax * ∂zwlhb
        Ay∂xwlhb = Ay * ∂xwlhb
        Ay∂zwlhb = Ay * ∂zwlhb
        Az∂xwlhb = Az * ∂xwlhb
        Az∂ywlhb = Az * ∂ywlhb

        ∇Ax = prefactor * (Ay∂zwlhb - Az∂ywlhb - curls_c[j][1])
        ∇Ay = prefactor * (Az∂xwlhb - Ax∂zwlhb - curls_c[j][2])
        ∇Az = prefactor * (Ax∂ywlhb - Ay∂xwlhb - curls_c[j][3])

        curls[j] = (∇Ax, ∇Ay, ∇Az)
    end

    scalars_out = ntuple(i -> scalars[i], Val(N))
    gradients_out = ntuple(i -> gradients[i], Val(G))
    divergences_out = ntuple(i -> divergences[i], Val(D))
    curls_out = ntuple(i -> curls[i], Val(C))

    output = (scalars_out, gradients_out, divergences_out, curls_out)
    return output
end

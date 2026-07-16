@inline function _general_quantity_interpolate_kernel(
    input :: InterpolationInput{2, T, V, Ktyp, NCOLUMN},
    reference_point :: NTuple{2,T},
    ha :: T,
    LBVH :: LinearBVH,
    catalog :: InterpolationCatalogConcise{2,N,G,Div,0}) :: Tuple{NTuple{N,T}, NTuple{G,NTuple{2,T}}, NTuple{Div,T}, Tuple{}} where {N, G, Div, T <: AbstractFloat, V <: AbstractVector{T}, Ktyp <: AbstractSPHKernel, NCOLUMN}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Kvalid = KernelFunctionValid(Ktyp, T)
    ShepardNormalization = catalog.scalar_snormalization
    x = get_xcoord(input)
    y = get_ycoord(input)
    @inbounds begin
        xa = reference_point[1]
        ya = reference_point[2]
    end

    # For aabb test
    radius = Kvalid * ha
    radius2 = radius * radius

    # Kernel quantities
    invha = inv(ha)
    invha2 = invha * invha
    prefactor = KernelFunctionnorm(Ktyp, Val(2), T) * invha2
    fdprefactor = prefactor * invha

    # Initialize counter
    ## Shepard Normalization
    S1 :: T = zero(T)

    ## Scalars
    scalars :: MVector{N, T} = zero(MVector{N, T})

    ## Gradients
    gradients_c :: MVector{G, SVector{2,T}} = MVector{G, SVector{2,T}}(ntuple(_ -> zero(SVector{2,T}), Val(G)))
    gradients_scalars :: MVector{G, T} = zero(MVector{G, T})

    ## Divergences
    divergences_c :: MVector{Div, T} = zero(MVector{Div, T})
    divergences_scalars :: MVector{Div, SVector{2,T}} = MVector{Div, SVector{2,T}}(ntuple(_ -> zero(SVector{2,T}), Val(Div)))

    ## Correction reduction
    correction_x :: T = zero(T)
    correction_y :: T = zero(T)

    # Traversal
    leaf_idx :: Int = zero(Int)
    p2leaf_d2 :: T   = zero(T)

    LinearBoundingVolumeHierarchy.@LBVH_gather_point_traversal LBVH reference_point radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δx = xa - x[leaf_idx]
            Δy = ya - y[leaf_idx]
            Δr = sqrt(p2leaf_d2)
            if iszero(Δr)
                Δx̂ = zero(T)
                Δŷ = zero(T)
            else
                invΔr = inv(Δr)
                Δx̂ = Δx * invΔr
                Δŷ = Δy * invΔr
            end

            q = Δr * invha

            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]

            # General volume element for standard SPH interpolation.
            mblρb = mb / ρb

            wb = K(q)
            ∂wb = KernelFunctionDiff(Ktyp, q)
            ∂xwb = ∂wb * Δx̂
            ∂ywb = ∂wb * Δŷ

            # Accumulate the dimensionless kernel shape first; the gather
            # prefactors are applied when constructing the outputs.
            mbwblρb = mblρb * wb
            mb∂xwblρb = mblρb * ∂xwb
            mb∂ywblρb = mblρb * ∂ywb

            correction_x += mb∂xwblρb
            correction_y += mb∂ywblρb

            # Shepard Normalization
            S1 += mbwblρb

            # Scalar interpolations
            @inbounds for j in 1:N
                Ab = input.quant[catalog.scalar_slots[j]][leaf_idx]
                scalars[j] += Ab * mbwblρb
            end

            # Gradient interpolations
            @inbounds for j in 1:G
                Ab = input.quant[catalog.grad_slots[j]][leaf_idx]

                Abmb∂xwblρb = Ab * mb∂xwblρb
                Abmb∂ywblρb = Ab * mb∂ywblρb

                gradients_c[j] += SVector{2,T}(Abmb∂xwblρb, Abmb∂ywblρb)
                gradients_scalars[j] += Ab * mbwblρb
            end

            # Divergence interpolations
            @inbounds for j in 1:Div
                Ax_slot, Ay_slot = catalog.div_slots[j]
                Axb = input.quant[Ax_slot][leaf_idx]
                Ayb = input.quant[Ay_slot][leaf_idx]

                divergences_c[j] += Axb * mb∂xwblρb + Ayb * mb∂ywblρb

                Axa = Axb * mbwblρb
                Aya = Ayb * mbwblρb
                divergences_scalars[j] += SVector{2,T}(Axa, Aya)
            end
        end
        #########################################################
    end

    # Preparing output
    if iszero(S1)
        scalars_out = ntuple(_ -> T(NaN), Val(N))
        gradients_out = ntuple(_ -> (T(NaN), T(NaN)), Val(G))
        divergences_out = ntuple(_ -> T(NaN), Val(Div))
        curls_out = ()
        output = (scalars_out, gradients_out, divergences_out, curls_out)
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
    gradients :: MVector{G, NTuple{2, T}} = MVector{G, NTuple{2, T}}(ntuple(_ -> (zero(T), zero(T)), Val(G)))
    divergences :: MVector{Div, T} = zero(MVector{Div, T})

    # Construct gradients
    @inbounds for j in 1:G
        A    = gradients_scalars[j] * invS1

        # Final result
        ∇Ax = fdprefactor * (gradients_c[j][1] - A * correction_x)
        ∇Ay = fdprefactor * (gradients_c[j][2] - A * correction_y)

        gradients[j] = (
            ∇Ax,
            ∇Ay,
        )
    end

    # Construct divergences
    @inbounds for j in 1:Div
        Ax   = divergences_scalars[j][1] * invS1
        Ay   = divergences_scalars[j][2] * invS1

        # Final result
        ∇A = fdprefactor * (divergences_c[j] - Ax * correction_x - Ay * correction_y)
        divergences[j] = ∇A
    end

    scalars_out = ntuple(i -> scalars[i], Val(N))
    gradients_out = ntuple(i -> gradients[i], Val(G))
    divergences_out = ntuple(i -> divergences[i], Val(Div))
    curls_out = ()
    output = (scalars_out, gradients_out, divergences_out, curls_out)
    return output
end

@inline function _general_quantity_interpolate_kernel(
    input :: InterpolationSmoothingVolumeInput{2, T, V, Ktyp, NCOLUMN},
    reference_point :: NTuple{2,T},
    ha :: T,
    LBVH :: LinearBVH,
    catalog :: InterpolationCatalogConcise{2,N,G,Div,0}) :: Tuple{NTuple{N,T}, NTuple{G,NTuple{2,T}}, NTuple{Div,T}, Tuple{}} where {N, G, Div, T <: AbstractFloat, V <: AbstractVector{T}, Ktyp <: AbstractSPHKernel, NCOLUMN}
    # Prepare for interpolation
    hfact = input.hfact
    η = hfact * hfact
    K = input.smoothed_kernel
    Kvalid = KernelFunctionValid(Ktyp, T)
    ShepardNormalization = catalog.scalar_snormalization
    x = get_xcoord(input)
    y = get_ycoord(input)
    @inbounds begin
        xa = reference_point[1]
        ya = reference_point[2]
    end

    # For aabb test
    radius = Kvalid * ha
    radius2 = radius * radius

    # Kernel quantities
    invha = inv(ha)
    invha2 = invha * invha
    invη = inv(η)
    baseprefactor = KernelFunctionnorm(Ktyp, Val(2), T) * invη
    sprefactor = baseprefactor * invha2
    fdprefactor = sprefactor * invha

    # Initialize counter
    ## Shepard Normalization
    S1 :: T = zero(T)

    ## Scalars
    scalars :: MVector{N, T} = zero(MVector{N, T})

    ## Gradients
    gradients_c :: MVector{G, SVector{2,T}} = MVector{G, SVector{2,T}}(ntuple(_ -> zero(SVector{2,T}), Val(G)))
    gradients_scalars :: MVector{G, T} = zero(MVector{G, T})

    ## Divergences
    divergences_c :: MVector{Div, T} = zero(MVector{Div, T})
    divergences_scalars :: MVector{Div, SVector{2,T}} = MVector{Div, SVector{2,T}}(ntuple(_ -> zero(SVector{2,T}), Val(Div)))

    ## Correction reduction
    correction_x :: T = zero(T)
    correction_y :: T = zero(T)

    # Traversal
    leaf_idx :: Int = zero(Int)
    p2leaf_d2 :: T   = zero(T)

    LinearBoundingVolumeHierarchy.@LBVH_gather_point_traversal LBVH reference_point radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δx = xa - x[leaf_idx]
            Δy = ya - y[leaf_idx]
            Δr = sqrt(p2leaf_d2)
            if iszero(Δr)
                Δx̂ = zero(T)
                Δŷ = zero(T)
            else
                invΔr = inv(Δr)
                Δx̂ = Δx * invΔr
                Δŷ = Δy * invΔr
            end

            q = Δr * invha

            wb = K(q)
            ∂wb = KernelFunctionDiff(Ktyp, q)
            ∂xwb = ∂wb * Δx̂
            ∂ywb = ∂wb * Δŷ

            hb = input.h[leaf_idx]
            hb2 = hb * hb
            hb2wb = hb2 * wb
            hb2∂xwb = hb2 * ∂xwb
            hb2∂ywb = hb2 * ∂ywb

            correction_x += hb2∂xwb
            correction_y += hb2∂ywb

            # Shepard Normalization
            S1 += hb2wb

            # Scalar interpolations
            @inbounds for j in 1:N
                Ab = input.quant[catalog.scalar_slots[j]][leaf_idx]
                scalars[j] += Ab * hb2wb
            end

            # Gradient interpolations
            @inbounds for j in 1:G
                Ab = input.quant[catalog.grad_slots[j]][leaf_idx]

                Abhb2∂xwb = Ab * hb2∂xwb
                Abhb2∂ywb = Ab * hb2∂ywb

                gradients_c[j] += SVector{2,T}(Abhb2∂xwb, Abhb2∂ywb)
                gradients_scalars[j] += Ab * hb2wb
            end

            # Divergence interpolations
            @inbounds for j in 1:Div
                Ax_slot, Ay_slot = catalog.div_slots[j]
                Axb = input.quant[Ax_slot][leaf_idx]
                Ayb = input.quant[Ay_slot][leaf_idx]

                divergences_c[j] += Axb * hb2∂xwb + Ayb * hb2∂ywb

                Axa = Axb * hb2wb
                Aya = Ayb * hb2wb
                divergences_scalars[j] += SVector{2,T}(Axa, Aya)
            end
        end
        #########################################################
    end

    # Preparing output
    if iszero(S1)
        scalars_out = ntuple(_ -> T(NaN), Val(N))
        gradients_out = ntuple(_ -> (T(NaN), T(NaN)), Val(G))
        divergences_out = ntuple(_ -> T(NaN), Val(Div))
        curls_out = ()
        output = (scalars_out, gradients_out, divergences_out, curls_out)
        return output
    end

    # Shepard normalization
    invS1 = inv(S1)
    @inbounds for j in 1:N
        if ShepardNormalization[j]
            scalars[j] *= invS1
        else
            scalars[j] *= sprefactor
        end
    end

    # Initialize output containers
    gradients :: MVector{G, NTuple{2, T}} = MVector{G, NTuple{2, T}}(ntuple(_ -> (zero(T), zero(T)), Val(G)))
    divergences :: MVector{Div, T} = zero(MVector{Div, T})

    # Construct gradients
    @inbounds for j in 1:G
        A    = gradients_scalars[j] * invS1

        # Final result
        ∇Ax = fdprefactor * (gradients_c[j][1] - A * correction_x)
        ∇Ay = fdprefactor * (gradients_c[j][2] - A * correction_y)

        gradients[j] = (∇Ax, ∇Ay)
    end

    # Construct divergences
    @inbounds for j in 1:Div
        Ax   = divergences_scalars[j][1] * invS1
        Ay   = divergences_scalars[j][2] * invS1

        # Final result
        ∇A = fdprefactor * (divergences_c[j] - Ax * correction_x - Ay * correction_y)
        divergences[j] = ∇A
    end

    scalars_out = ntuple(i -> scalars[i], Val(N))
    gradients_out = ntuple(i -> gradients[i], Val(G))
    divergences_out = ntuple(i -> divergences[i], Val(Div))
    curls_out = ()
    output = (scalars_out, gradients_out, divergences_out, curls_out)
    return output
end

@inline function _general_quantity_interpolate_kernel(
                        input :: InterpolationInput{3, T, V, Ktyp, NCOLUMN},
                        reference_point :: NTuple{3,T},
                        ha :: T,
                        LBVH :: LinearBVH,
                        catalog :: InterpolationCatalogConcise{3,N,G,D,C}) :: Tuple{NTuple{N,T}, NTuple{G,NTuple{3,T}}, NTuple{D,T}, NTuple{C,NTuple{3,T}}} where {N, G, D, C, T <: AbstractFloat, V <: AbstractVector{T}, Ktyp <: AbstractSPHKernel, NCOLUMN}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Kvalid = KernelFunctionValid(Ktyp, T)
    ShepardNormalization = catalog.scalar_snormalization
    x = get_xcoord(input)
    y = get_ycoord(input)
    z = get_zcoord(input)
    @inbounds begin
        xa = reference_point[1]; ya = reference_point[2]; za = reference_point[3];
    end

    # For aabb test
    radius = Kvalid * ha
    radius2 = radius * radius

    # Kernel quantities
    ## Gather uses the query smoothing length `ha`, so the h-scaling is common
    ## to all neighbours and can be applied after the dimensionless reductions.
    invha = inv(ha)
    invha3 = invha * invha * invha
    prefactor = KernelFunctionnorm(Ktyp, Val(3), T) * invha3
    fdprefactor = prefactor * invha

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
    ## This is the kernel-gradient term used by the Shepard-consistent
    ## derivative correction. Keep it as scalar components for GPU kernels.
    correction_x :: T = zero(T)
    correction_y :: T = zero(T)
    correction_z :: T = zero(T)

    # Traversal
    leaf_idx :: Int = zero(Int)
    p2leaf_d2 :: T   = zero(T)

    LinearBoundingVolumeHierarchy.@LBVH_gather_point_traversal LBVH reference_point radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            xb = x[leaf_idx]; yb = y[leaf_idx]; zb = z[leaf_idx]
            Δx = xa - xb
            Δy = ya - yb
            Δz = za - zb

            Δr = sqrt(p2leaf_d2)
            q  = Δr / ha
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

            wb = K(q)
            ∂wb = KernelFunctionDiff(Ktyp, q)
            ∂xwb = ∂wb * Δx̂
            ∂ywb = ∂wb * Δŷ
            ∂zwb = ∂wb * Δẑ

            # Accumulate the dimensionless kernel shape first; the gather
            # prefactors Cnorm / ha^3 and Cnorm / ha^4 are applied at output.
            mbwblρb = mblρb * wb
            mb∂xwblρb = mblρb * ∂xwb
            mb∂ywblρb = mblρb * ∂ywb
            mb∂zwblρb = mblρb * ∂zwb

            correction_x += mb∂xwblρb
            correction_y += mb∂ywblρb
            correction_z += mb∂zwblρb

            # Shepard Normalization
            S1 += mbwblρb

            # Scalar interpolations
            @inbounds for j in 1:N
                slot = catalog.scalar_slots[j]
                Ab = input.quant[slot][leaf_idx]
                scalars[j] += Ab * mbwblρb
            end

            # Gradient interpolations
            @inbounds for j in 1:G
                slot = catalog.grad_slots[j]
                Ab = input.quant[slot][leaf_idx]

                Abmb∂xwblρb = Ab * mb∂xwblρb
                Abmb∂ywblρb = Ab * mb∂ywblρb
                Abmb∂zwblρb = Ab * mb∂zwblρb

                gradients_c[j] += SVector{3,T}(Abmb∂xwblρb, Abmb∂ywblρb, Abmb∂zwblρb)
                gradients_scalars[j] += Ab * mbwblρb
            end

            # Divergence interpolations
            @inbounds for j in 1:D
                slot = catalog.div_slots[j]
                Ax_column_idx, Ay_column_idx, Az_column_idx = slot
                Axb = input.quant[Ax_column_idx][leaf_idx]
                Ayb = input.quant[Ay_column_idx][leaf_idx]
                Azb = input.quant[Az_column_idx][leaf_idx]

                divergences_c[j] += Axb * mb∂xwblρb + Ayb * mb∂ywblρb + Azb * mb∂zwblρb

                Axa = Axb * mbwblρb
                Aya = Ayb * mbwblρb
                Aza = Azb * mbwblρb
                divergences_scalars[j] += SVector{3,T}(Axa, Aya, Aza)
            end

            # Curl interpolations
            @inbounds for j in 1:C
                slot = catalog.curl_slots[j]
                Ax_column_idx, Ay_column_idx, Az_column_idx = slot
                Axb = input.quant[Ax_column_idx][leaf_idx]
                Ayb = input.quant[Ay_column_idx][leaf_idx]
                Azb = input.quant[Az_column_idx][leaf_idx]

                Axbmb∂ywblρb = Axb * mb∂ywblρb
                Axbmb∂zwblρb = Axb * mb∂zwblρb
                Aybmb∂xwblρb = Ayb * mb∂xwblρb
                Aybmb∂zwblρb = Ayb * mb∂zwblρb
                Azbmb∂xwblρb = Azb * mb∂xwblρb
                Azbmb∂ywblρb = Azb * mb∂ywblρb

                curls_c[j] += SVector{3,T}(Aybmb∂zwblρb - Azbmb∂ywblρb, Azbmb∂xwblρb - Axbmb∂zwblρb, Axbmb∂ywblρb - Aybmb∂xwblρb)

                Axa = Axb * mbwblρb
                Aya = Ayb * mbwblρb
                Aza = Azb * mbwblρb
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
        ∇Ax = fdprefactor * (gradients_c[j][1] - A * correction_x)
        ∇Ay = fdprefactor * (gradients_c[j][2] - A * correction_y)
        ∇Az = fdprefactor * (gradients_c[j][3] - A * correction_z)

        gradients[j] = (∇Ax, ∇Ay, ∇Az)
    end

    # Construct divergences
    @inbounds for j in 1:D
        Ax   = divergences_scalars[j][1] * invS1
        Ay   = divergences_scalars[j][2] * invS1
        Az   = divergences_scalars[j][3] * invS1

        # Final result
        ∇A = fdprefactor * (divergences_c[j] - Ax * correction_x - Ay * correction_y - Az * correction_z)
        divergences[j] = ∇A
    end

    # Construct curls
    @inbounds for j in 1:C
        Ax   = curls_scalars[j][1] * invS1
        Ay   = curls_scalars[j][2] * invS1
        Az   = curls_scalars[j][3] * invS1

        m∂xwlρ = correction_x
        m∂ywlρ = correction_y
        m∂zwlρ = correction_z

        # Final result
        Axm∂ywlρ = Ax * m∂ywlρ
        Axm∂zwlρ = Ax * m∂zwlρ
        Aym∂xwlρ = Ay * m∂xwlρ
        Aym∂zwlρ = Ay * m∂zwlρ
        Azm∂xwlρ = Az * m∂xwlρ
        Azm∂ywlρ = Az * m∂ywlρ

        ∇Ax = fdprefactor * (Aym∂zwlρ - Azm∂ywlρ - curls_c[j][1])
        ∇Ay = fdprefactor * (Azm∂xwlρ - Axm∂zwlρ - curls_c[j][2])
        ∇Az = fdprefactor * (Axm∂ywlρ - Aym∂xwlρ - curls_c[j][3])

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
                        ha :: T,
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
    # For gradients
    ## Gather uses the query smoothing length `ha`, so these powers are common
    ## to all neighbours and can be applied after the raw reductions.
    invha3 = inv(ha * ha * ha)
    invha4 = inv(ha * ha * ha * ha)

    # Prefactor for all interpolation (hfact^-3 Cnorm)
    ## The smoothing-volume relation replaces m_b / rho_b with h_b^3 / hfact^3.
    ## The remaining common factor is Cnorm / hfact^3.
    invη = inv(η)
    prefactor = KernelFunctionnorm(Ktyp, Val(3), T) * invη
    sprefactor = prefactor * invha3
    fdprefactor = prefactor * invha4

    # For aabb test
    radius = Kvalid * ha
    radius2 = radius * radius

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
    ## Equivalent to the standard correction with m_b/rho_b -> h_b^3/hfact^3;
    ## the common hfact and kernel-normalization factors are applied later.
    correction_x :: T = zero(T)
    correction_y :: T = zero(T)
    correction_z :: T = zero(T)

    # Traversal
    leaf_idx :: Int = zero(Int)
    p2leaf_d2 :: T   = zero(T)

    LinearBoundingVolumeHierarchy.@LBVH_gather_point_traversal LBVH reference_point radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            xb = x[leaf_idx]; yb = y[leaf_idx]; zb = z[leaf_idx]
            Δx = xa - xb
            Δy = ya - yb
            Δz = za - zb

            Δr = sqrt(p2leaf_d2)
            q  = Δr / ha
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

            # Gather evaluates w(Delta r / ha), while the smoothing-volume
            # relation supplies the neighbour volume h_b^3 / hfact^3.
            hb = input.h[leaf_idx]
            hb3 = hb * hb * hb

            wb = K(q)
            ∂wb = KernelFunctionDiff(Ktyp, q)
            ∂xwb = ∂wb * Δx̂
            ∂ywb = ∂wb * Δŷ
            ∂zwb = ∂wb * Δẑ

            # Raw dimensionless reductions; multiply by Cnorm / hfact^3 and
            # the gather powers of ha only when constructing final outputs.
            hb3wb = hb3 * wb
            hb3∂xwb = hb3 * ∂xwb
            hb3∂ywb = hb3 * ∂ywb
            hb3∂zwb = hb3 * ∂zwb

            correction_x += hb3∂xwb
            correction_y += hb3∂ywb
            correction_z += hb3∂zwb

            # Shepard Normalization
            S1b = hb3 * wb
            S1 += S1b

            # Scalar interpolations
            @inbounds for j in 1:N
                slot = catalog.scalar_slots[j]
                Ab = input.quant[slot][leaf_idx]
                scalars[j] += Ab * hb3wb
            end

            # Gradient interpolations
            @inbounds for j in 1:G
                slot = catalog.grad_slots[j]
                Ab = input.quant[slot][leaf_idx]

                Abhb3∂xwb = Ab * hb3∂xwb
                Abhb3∂ywb = Ab * hb3∂ywb
                Abhb3∂zwb = Ab * hb3∂zwb

                gradients_c[j] += SVector{3,T}(Abhb3∂xwb, Abhb3∂ywb, Abhb3∂zwb)
                gradients_scalars[j] += Ab * hb3wb
            end

            # Divergence interpolations
            @inbounds for j in 1:D
                slot = catalog.div_slots[j]
                Ax_column_idx, Ay_column_idx, Az_column_idx = slot
                Axb = input.quant[Ax_column_idx][leaf_idx]
                Ayb = input.quant[Ay_column_idx][leaf_idx]
                Azb = input.quant[Az_column_idx][leaf_idx]

                divergences_c[j] += Axb * hb3∂xwb + Ayb * hb3∂ywb + Azb * hb3∂zwb

                Axa = Axb * hb3wb
                Aya = Ayb * hb3wb
                Aza = Azb * hb3wb
                divergences_scalars[j] += SVector{3,T}(Axa, Aya, Aza)
            end

            # Curl interpolations
            @inbounds for j in 1:C
                slot = catalog.curl_slots[j]
                Ax_column_idx, Ay_column_idx, Az_column_idx = slot
                Axb = input.quant[Ax_column_idx][leaf_idx]
                Ayb = input.quant[Ay_column_idx][leaf_idx]
                Azb = input.quant[Az_column_idx][leaf_idx]

                Axbhb3∂ywb = Axb * hb3∂ywb
                Axbhb3∂zwb = Axb * hb3∂zwb
                Aybhb3∂xwb = Ayb * hb3∂xwb
                Aybhb3∂zwb = Ayb * hb3∂zwb
                Azbhb3∂xwb = Azb * hb3∂xwb
                Azbhb3∂ywb = Azb * hb3∂ywb

                curls_c[j] += SVector{3,T}(Aybhb3∂zwb - Azbhb3∂ywb, Azbhb3∂xwb - Axbhb3∂zwb, Axbhb3∂ywb - Aybhb3∂xwb)

                Axa = Axb * hb3wb
                Aya = Ayb * hb3wb
                Aza = Azb * hb3wb
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
            scalars[j] *= sprefactor
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
        ∇Ax = fdprefactor * (gradients_c[j][1] - A * correction_x)
        ∇Ay = fdprefactor * (gradients_c[j][2] - A * correction_y)
        ∇Az = fdprefactor * (gradients_c[j][3] - A * correction_z)

        gradients[j] = (∇Ax, ∇Ay, ∇Az)
    end

    # Construct divergences
    @inbounds for j in 1:D
        Ax   = divergences_scalars[j][1] * invS1
        Ay   = divergences_scalars[j][2] * invS1
        Az   = divergences_scalars[j][3] * invS1

        # Final result
        ∇A = fdprefactor * (divergences_c[j] - Ax * correction_x - Ay * correction_y - Az * correction_z)
        divergences[j] = ∇A
    end

    # Construct curls
    @inbounds for j in 1:C
        Ax   = curls_scalars[j][1] * invS1
        Ay   = curls_scalars[j][2] * invS1
        Az   = curls_scalars[j][3] * invS1

        h3∂xw = correction_x
        h3∂yw = correction_y
        h3∂zw = correction_z

        # Final result
        Axh3∂yw = Ax * h3∂yw
        Axh3∂zw = Ax * h3∂zw
        Ayh3∂xw = Ay * h3∂xw
        Ayh3∂zw = Ay * h3∂zw
        Azh3∂xw = Az * h3∂xw
        Azh3∂yw = Az * h3∂yw

        ∇Ax = fdprefactor * (Ayh3∂zw - Azh3∂yw - curls_c[j][1])
        ∇Ay = fdprefactor * (Azh3∂xw - Axh3∂zw - curls_c[j][2])
        ∇Az = fdprefactor * (Axh3∂yw - Ayh3∂xw - curls_c[j][3])

        curls[j] = (∇Ax, ∇Ay, ∇Az)
    end

    scalars_out = ntuple(i -> scalars[i], Val(N))
    gradients_out = ntuple(i -> gradients[i], Val(G))
    divergences_out = ntuple(i -> divergences[i], Val(D))
    curls_out = ntuple(i -> curls[i], Val(C))

    output = (scalars_out, gradients_out, divergences_out, curls_out)
    return output
end

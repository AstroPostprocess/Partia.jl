@inline function _gradient_density_kernel(input :: InterpolationInput{3, T}, reference_point :: NTuple{3, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpGather}) where {T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)
    x = get_xcoord(input)
    y = get_ycoord(input)
    z = get_zcoord(input)

    radius = Kvalid * ha
    radius2 = radius * radius

    ∇ρxf :: T = zero(T)
    ∇ρyf :: T = zero(T)
    ∇ρzf :: T = zero(T)
    ∇ρxb :: T = zero(T)
    ∇ρyb :: T = zero(T)
    ∇ρzb :: T = zero(T)
    ρ :: T = zero(T)


    # Traversal
    leaf_idx :: Int = zero(Int)
    p2leaf_d2 :: T   = zero(T)

    LinearBoundingVolumeHierarchy.@LBVH_gather_point_traversal LBVH reference_point radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (x[leaf_idx], y[leaf_idx], z[leaf_idx])
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]

            ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, ha, K)
            ρ += _density_accumulation(reference_point, rb, mb, ha, K)
            ∇ρxf += ∇ρxfW
            ∇ρyf += ∇ρyfW
            ∇ρzf += ∇ρzfW
            ∇ρxb += ∇ρxbW
            ∇ρyb += ∇ρybW
            ∇ρzb += ∇ρzbW
        end
        #########################################################
    end

    if iszero(ρ)
        return (T(NaN), T(NaN), T(NaN))
    end

    ∇ρxb *= ρ
    ∇ρyb *= ρ
    ∇ρzb *= ρ

    ∇ρx = (∇ρxf - ∇ρxb)
    ∇ρy = (∇ρyf - ∇ρyb)
    ∇ρz = (∇ρzf - ∇ρzb)
    return (∇ρx, ∇ρy, ∇ρz)
end

@inline function _gradient_density_kernel(input :: InterpolationInput{3, T}, reference_point :: NTuple{3, T}, LBVH :: LinearBVH, :: Type{itpScatter}) where {T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)
    x = get_xcoord(input)
    y = get_ycoord(input)
    z = get_zcoord(input)

    ∇ρxf :: T = zero(T)
    ∇ρyf :: T = zero(T)
    ∇ρzf :: T = zero(T)
    ∇ρxb :: T = zero(T)
    ∇ρyb :: T = zero(T)
    ∇ρzb :: T = zero(T)
    ρ :: T = zero(T)


    # Traversal
    leaf_idx :: Int = zero(Int)
    p2leaf_d2 :: T   = zero(T)
    hb :: T   = zero(T)

    LinearBoundingVolumeHierarchy.@LBVH_scatter_point_traversal LBVH reference_point Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (x[leaf_idx], y[leaf_idx], z[leaf_idx])
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]

            ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, hb, K)
            ρ += _density_accumulation(reference_point, rb, mb, hb, K)
            ∇ρxf += ∇ρxfW
            ∇ρyf += ∇ρyfW
            ∇ρzf += ∇ρzfW
            ∇ρxb += ∇ρxbW
            ∇ρyb += ∇ρybW
            ∇ρzb += ∇ρzbW
        end
        #########################################################
    end

    if iszero(ρ)
        return (T(NaN), T(NaN), T(NaN))
    end

    ∇ρxb *= ρ
    ∇ρyb *= ρ
    ∇ρzb *= ρ

    ∇ρx = (∇ρxf - ∇ρxb)
    ∇ρy = (∇ρyf - ∇ρyb)
    ∇ρz = (∇ρzf - ∇ρzb)
    return (∇ρx, ∇ρy, ∇ρz)
end

@inline function _gradient_density_kernel(input :: InterpolationInput{3, T}, reference_point :: NTuple{3, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpSymmetric}) where {T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)
    x = get_xcoord(input)
    y = get_ycoord(input)
    z = get_zcoord(input)

    radius = Kvalid * ha
    radius2 = radius * radius

    ∇ρxf :: T = zero(T)
    ∇ρyf :: T = zero(T)
    ∇ρzf :: T = zero(T)
    ∇ρxb :: T = zero(T)
    ∇ρyb :: T = zero(T)
    ∇ρzb :: T = zero(T)
    ρ :: T = zero(T)


    # Traversal
    leaf_idx :: Int = zero(Int)
    p2leaf_d2 :: T   = zero(T)
    hb :: T   = zero(T)

    LinearBoundingVolumeHierarchy.@LBVH_symmetric_point_traversal LBVH reference_point Kvalid radius2 leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (x[leaf_idx], y[leaf_idx], z[leaf_idx])
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]

            ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
            ρ += _density_accumulation(reference_point, rb, mb, ha, hb, K)
            ∇ρxf += ∇ρxfW
            ∇ρyf += ∇ρyfW
            ∇ρzf += ∇ρzfW
            ∇ρxb += ∇ρxbW
            ∇ρyb += ∇ρybW
            ∇ρzb += ∇ρzbW
        end
        #########################################################
    end

    if iszero(ρ)
        return (T(NaN), T(NaN), T(NaN))
    end

    ∇ρxb *= ρ
    ∇ρyb *= ρ
    ∇ρzb *= ρ

    ∇ρx = (∇ρxf - ∇ρxb)
    ∇ρy = (∇ρyf - ∇ρyb)
    ∇ρz = (∇ρzf - ∇ρzb)
    return (∇ρx, ∇ρy, ∇ρz)
end

@inline function _gradient_quantity_interpolate_kernel(input :: InterpolationInput{3, T}, reference_point :: NTuple{3, T}, ha :: T, LBVH :: LinearBVH, column_idx :: Int, :: Type{itpGather}) where {T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)
    x = get_xcoord(input)
    y = get_ycoord(input)
    z = get_zcoord(input)

    radius = Kvalid * ha
    radius2 = radius * radius

    ∇Axf :: T = zero(T)
    ∇Ayf :: T = zero(T)
    ∇Azf :: T = zero(T)
    ∇Axb :: T = zero(T)
    ∇Ayb :: T = zero(T)
    ∇Azb :: T = zero(T)
    A :: T = zero(T)
    S1 :: T = zero(T)



    # Traversal
    leaf_idx :: Int = zero(Int)
    p2leaf_d2 :: T   = zero(T)

    LinearBoundingVolumeHierarchy.@LBVH_gather_point_traversal LBVH reference_point radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (x[leaf_idx], y[leaf_idx], z[leaf_idx])
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]
            Ab = input.quant[column_idx][leaf_idx]

            ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)
            ∇Axf += ∇AxfW
            ∇Ayf += ∇AyfW
            ∇Azf += ∇AzfW
            ∇Axb += ∇AxbW
            ∇Ayb += ∇AybW
            ∇Azb += ∇AzbW
            A += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)

            S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, K)
            S1 += S1b

        end
        #########################################################
    end
    iszero(S1) && return (T(NaN), T(NaN), T(NaN))

    A /= S1
    ∇Axb *= A
    ∇Ayb *= A
    ∇Azb *= A

    ∇Ax = (∇Axf - ∇Axb)
    ∇Ay = (∇Ayf - ∇Ayb)
    ∇Az = (∇Azf - ∇Azb)


    return (∇Ax, ∇Ay, ∇Az)
end

@inline function _gradient_quantity_interpolate_kernel(input :: InterpolationInput{3, T}, reference_point :: NTuple{3, T}, LBVH :: LinearBVH, column_idx :: Int, :: Type{itpScatter}) where {T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)
    x = get_xcoord(input)
    y = get_ycoord(input)
    z = get_zcoord(input)

    ∇Axf :: T = zero(T)
    ∇Ayf :: T = zero(T)
    ∇Azf :: T = zero(T)
    ∇Axb :: T = zero(T)
    ∇Ayb :: T = zero(T)
    ∇Azb :: T = zero(T)
    A :: T = zero(T)
    S1 :: T = zero(T)



    # Traversal
    leaf_idx :: Int = zero(Int)
    p2leaf_d2 :: T   = zero(T)
    hb :: T   = zero(T)

    LinearBoundingVolumeHierarchy.@LBVH_scatter_point_traversal LBVH reference_point Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (x[leaf_idx], y[leaf_idx], z[leaf_idx])
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]
            Ab = input.quant[column_idx][leaf_idx]

            ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)
            ∇Axf += ∇AxfW
            ∇Ayf += ∇AyfW
            ∇Azf += ∇AzfW
            ∇Axb += ∇AxbW
            ∇Ayb += ∇AybW
            ∇Azb += ∇AzbW
            A += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)

            S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hb, K)
            S1 += S1b

        end
        #########################################################
    end
    iszero(S1) && return (T(NaN), T(NaN), T(NaN))

    A /= S1
    ∇Axb *= A
    ∇Ayb *= A
    ∇Azb *= A

    ∇Ax = (∇Axf - ∇Axb)
    ∇Ay = (∇Ayf - ∇Ayb)
    ∇Az = (∇Azf - ∇Azb)


    return (∇Ax, ∇Ay, ∇Az)
end

@inline function _gradient_quantity_interpolate_kernel(input :: InterpolationInput{3, T}, reference_point :: NTuple{3, T}, ha :: T, LBVH :: LinearBVH, column_idx :: Int, :: Type{itpSymmetric}) where {T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)
    x = get_xcoord(input)
    y = get_ycoord(input)
    z = get_zcoord(input)

    radius = Kvalid * ha
    radius2 = radius * radius

    ∇Axf :: T = zero(T)
    ∇Ayf :: T = zero(T)
    ∇Azf :: T = zero(T)
    ∇Axb :: T = zero(T)
    ∇Ayb :: T = zero(T)
    ∇Azb :: T = zero(T)
    A :: T = zero(T)
    S1 :: T = zero(T)



    # Traversal
    leaf_idx :: Int = zero(Int)
    p2leaf_d2 :: T   = zero(T)
    hb :: T   = zero(T)

    LinearBoundingVolumeHierarchy.@LBVH_symmetric_point_traversal LBVH reference_point Kvalid radius2 leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (x[leaf_idx], y[leaf_idx], z[leaf_idx])
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]
            Ab = input.quant[column_idx][leaf_idx]

            ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)
            ∇Axf += ∇AxfW
            ∇Ayf += ∇AyfW
            ∇Azf += ∇AzfW
            ∇Axb += ∇AxbW
            ∇Ayb += ∇AybW
            ∇Azb += ∇AzbW
            A += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)

            S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
            S1 += S1b

        end
        #########################################################
    end
    iszero(S1) && return (T(NaN), T(NaN), T(NaN))

    A /= S1
    ∇Axb *= A
    ∇Ayb *= A
    ∇Azb *= A

    ∇Ax = (∇Axf - ∇Axb)
    ∇Ay = (∇Ayf - ∇Ayb)
    ∇Az = (∇Azf - ∇Azb)


    return (∇Ax, ∇Ay, ∇Az)
end

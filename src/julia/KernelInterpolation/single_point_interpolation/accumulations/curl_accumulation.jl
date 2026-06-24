"""
∇×A(r) = -∑_b m_b/ρ_b*(A_b-A(r))×∇W(r-r_b)
       = -(∑_b m_b/ρ_b*A_b×∇W(r-r_b)) + A(r)×(∑_b m_b/ρ_b*∇W(r-r_b))
       = -(∇×Af - ∇×Ab)
"""
# Single column curl value intepolation
@inline function _curl_quantity_accumulation(Δx :: T, Δy :: T, Δz :: T, mb :: T, ρb :: T, Axb :: T, Ayb :: T, Azb :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    ∇W = Smoothed_gradient_kernel_function(Ktyp, Δx, Δy, Δz, h)
    ∂xW = ∇W[1]
    ∂yW = ∇W[2]
    ∂zW = ∇W[3]

    invρb = inv(ρb)

    # Gradient
    mblρb∂xW = mb * invρb * ∂xW
    mblρb∂yW = mb * invρb * ∂yW
    mblρb∂zW = mb * invρb * ∂zW

    ∇Axf = Ayb * mblρb∂zW -  Azb * mblρb∂yW
    ∇Ayf = Azb * mblρb∂xW -  Axb * mblρb∂zW
    ∇Azf = Axb * mblρb∂yW -  Ayb * mblρb∂xW

    return ∇Axf, ∇Ayf, ∇Azf, mblρb∂xW, mblρb∂yW, mblρb∂zW
end

@inline function _curl_quantity_accumulation(ra :: NTuple{D, T}, rb :: NTuple{D, T}, mb :: T, ρb :: T, Axb :: T, Ayb :: T, Azb :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel, D}
    Ktyp = typeof(smoothed_kernel)
    ∇W = Smoothed_gradient_kernel_function(Ktyp, ra, rb, h)
    ∂xW = ∇W[1]
    ∂yW = ∇W[2]
    ∂zW = ∇W[3]

    invρb = inv(ρb)

    # Gradient
    mblρb∂xW = mb * invρb * ∂xW
    mblρb∂yW = mb * invρb * ∂yW
    mblρb∂zW = mb * invρb * ∂zW

    ∇Axf = Ayb * mblρb∂zW -  Azb * mblρb∂yW
    ∇Ayf = Azb * mblρb∂xW -  Axb * mblρb∂zW
    ∇Azf = Axb * mblρb∂yW -  Ayb * mblρb∂xW

    return ∇Axf, ∇Ayf, ∇Azf, mblρb∂xW, mblρb∂yW, mblρb∂zW
end

######################################################################################

# Dispatch tag for constructing grid

######################################################################################
abstract type AbstractCoordinateSystem end

struct Cartesian <: AbstractCoordinateSystem end
struct Polar <: AbstractCoordinateSystem end        # (s, ϕ)
struct Cylindrical <: AbstractCoordinateSystem end        # (s, ϕ, z)
struct Spherical <: AbstractCoordinateSystem end        # (r, ϕ, θ)


@inline function _coordinate_grid_isapprox(
    actual :: NTuple{D,VA},
    expected :: NTuple{D,VE};
    atol :: Real = 1.0e-8,
    rtol :: Real = 1.0e-8,
) where {D, T <: AbstractFloat,VA <: AbstractVector{T}, VE <: AbstractVector{T}}
    @inbounds for d in 1:D
        length(actual[d]) == length(expected[d]) || return false
        for i in eachindex(actual[d], expected[d])
            isapprox(actual[d][i], expected[d][i]; atol = atol, rtol = rtol) || return false
        end
    end
    return true
end

function _cartesian_plane_coordinates(frame :: Frame{TF}, width :: TF, height :: TF, nx :: TI, ny :: TI) where {TF <: AbstractFloat, TI <: Integer}
    # Validate the physical extent and sampling resolution
    width > zero(TF) ||
        throw(ArgumentError("width must be positive."))

    height > zero(TF) ||
        throw(ArgumentError("height must be positive."))

    nx >= TI(2) ||
        throw(ArgumentError("nx must be at least 2."))

    ny >= TI(2) ||
        throw(ArgumentError("ny must be at least 2."))

    # Get the current plane centre and in-plane basis vectors
    x0, y0, z0 = frame_position(frame)
    rx, ry, rz = frame_right(frame)
    ux, uy, uz = frame_up(frame)

    # Include both boundaries so that the sampled extents are exactly
    # `width` along the local right direction and `height` along the local up direction
    Δξ = width  / TF(nx - 1)
    Δu = height / TF(ny - 1)

    ξmin = -width  / TF(2)
    umin = -height / TF(2)

    # Allocate Cartesian coordinates in structure-of-arrays form
    N = nx * ny

    x = Vector{TF}(undef, N)
    y = Vector{TF}(undef, N)
    z = Vector{TF}(undef, N)

    @inbounds for j in 1:ny
        η = umin + TF(j - 1) * Δu

        @simd for i in 1:nx
            ξ = ξmin + TF(i - 1) * Δξ
            k = i + (j - 1) * nx

            # Map local plane coordinates to global Cartesian coordinates
            x[k] = x0 + ξ * rx + η * ux
            y[k] = y0 + ξ * ry + η * uy
            z[k] = z0 + ξ * rz + η * uz
        end
    end

    return x, y, z
end

function _polar_plane_coordinates(frame :: Frame{TF}, smin :: TF, smax :: TF, ns :: TI, nϕ :: TI) where {TF <: AbstractFloat, TI <: Integer}
    smin >= zero(TF) ||
        throw(ArgumentError("smin must be nonnegative."))

    smax > smin ||
        throw(ArgumentError("smax must be greater than smin."))

    ns >= TI(2) ||
        throw(ArgumentError("ns must be at least 2."))

    nϕ >= TI(3) ||
        throw(ArgumentError("nϕ must be at least 3."))

    # Get the current plane centre and in-plane basis vectors
    x0, y0, z0 = frame_position(frame)
    rx, ry, rz = frame_right(frame)
    ux, uy, uz = frame_up(frame)

    # Include both s-coordinate boundaries
    Δs = (smax - smin) / TF(ns - 1)

    # Sample the periodic angular direction without duplicating ϕ = 2π
    Δϕ = TF(2π) / TF(nϕ)

    N = ns * nϕ

    x = Vector{TF}(undef, N)
    y = Vector{TF}(undef, N)
    z = Vector{TF}(undef, N)

    @inbounds for j in 1:nϕ
        ϕ = TF(j - 1) * Δϕ
        sinϕ, cosϕ = sincos(ϕ)

        @simd for i in 1:ns
            s = smin + TF(i - 1) * Δs
            k = i + (j - 1) * ns

            ξ = s * cosϕ
            η = s * sinϕ

            x[k] = x0 + ξ * rx + η * ux
            y[k] = y0 + ξ * ry + η * uy
            z[k] = z0 + ξ * rz + η * uz
        end
    end

    return x, y, z
end

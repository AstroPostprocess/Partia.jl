######################################################################################

# Dispatch tag for constructing grid

######################################################################################
"""
    AxisParam{TF} = Tuple{TF, TF, Int}

Axis specification tuple `(xmin, xmax, xn)`.

# Type Parameters
- `TF <: AbstractFloat` : Floating-point type for axis endpoints.

# Fields / Layout
- `xmin :: TF` : Axis minimum.
- `xmax :: TF` : Axis maximum.
- `xn :: Int`  : Number of points.
"""
const AxisParam{TF} = Tuple{TF, TF, Int}

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

function _cartesian_plane_coordinates(frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    xmin, xmax, nx = xparams
    ymin, ymax, ny = yparams

    xmax > xmin ||
        throw(ArgumentError("xmax must be greater than xmin."))

    ymax > ymin ||
        throw(ArgumentError("ymax must be greater than ymin."))

    nx >= 2 ||
        throw(ArgumentError("nx must be at least 2."))

    ny >= 2 ||
        throw(ArgumentError("ny must be at least 2."))

    # Get the current plane centre and in-plane basis vectors
    x0, y0, z0 = frame_position(frame)
    rx, ry, rz = frame_right(frame)
    ux, uy, uz = frame_up(frame)

    # Include both Cartesian plane boundaries
    Δx = (xmax - xmin) / TF(nx - 1)
    Δy = (ymax - ymin) / TF(ny - 1)

    # Allocate Cartesian coordinates in structure-of-arrays form
    N = nx * ny

    x = Vector{TF}(undef, N)
    y = Vector{TF}(undef, N)
    z = Vector{TF}(undef, N)

    @inbounds for j in 1:ny
        η = ymin + TF(j - 1) * Δy

        @simd for i in 1:nx
            ξ = xmin + TF(i - 1) * Δx
            k = i + (j - 1) * nx

            # Map local plane coordinates to global Cartesian coordinates
            x[k] = x0 + ξ * rx + η * ux
            y[k] = y0 + ξ * ry + η * uy
            z[k] = z0 + ξ * rz + η * uz
        end
    end

    return x, y, z
end

function _cartesian_box_coordinates(frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}, zparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    xmin, xmax, nx = xparams
    ymin, ymax, ny = yparams
    zmin, zmax, nz = zparams

    xmax > xmin ||
        throw(ArgumentError("xmax must be greater than xmin."))

    ymax > ymin ||
        throw(ArgumentError("ymax must be greater than ymin."))

    zmax > zmin ||
        throw(ArgumentError("zmax must be greater than zmin."))

    nx >= 2 ||
        throw(ArgumentError("nx must be at least 2."))

    ny >= 2 ||
        throw(ArgumentError("ny must be at least 2."))

    nz >= 2 ||
        throw(ArgumentError("nz must be at least 2."))

    # Get the current box centre and local basis vectors
    x0, y0, z0 = frame_position(frame)
    rx, ry, rz = frame_right(frame)
    ux, uy, uz = frame_up(frame)
    fx, fy, fz = frame_forward(frame)

    # Include all Cartesian box boundaries
    Δx = (xmax - xmin) / TF(nx - 1)
    Δy = (ymax - ymin) / TF(ny - 1)
    Δz = (zmax - zmin) / TF(nz - 1)

    N = nx * ny * nz

    x = Vector{TF}(undef, N)
    y = Vector{TF}(undef, N)
    z = Vector{TF}(undef, N)

    @inbounds for k3 in 1:nz
        ζ = zmin + TF(k3 - 1) * Δz

        for j in 1:ny
            η = ymin + TF(j - 1) * Δy

            @simd for i in 1:nx
                ξ = xmin + TF(i - 1) * Δx
                k = i + (j - 1) * nx + (k3 - 1) * nx * ny

                x[k] = x0 + ξ * rx + η * ux + ζ * fx
                y[k] = y0 + ξ * ry + η * uy + ζ * fy
                z[k] = z0 + ξ * rz + η * uz + ζ * fz
            end
        end
    end

    return x, y, z
end

function _polar_plane_coordinates(frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    smin, smax, ns = sparams
    ϕmin, ϕmax, nϕ = ϕparams

    smin >= zero(TF) ||
        throw(ArgumentError("smin must be nonnegative."))

    smax > smin ||
        throw(ArgumentError("smax must be greater than smin."))

    ϕmin >= zero(TF) && (ϕmax <= TF(2π) || isapprox(ϕmax, TF(2π))) && ϕmax > ϕmin ||
        throw(ArgumentError("angular range must satisfy 0 ≤ ϕmin < ϕmax ≤ 2π."))

    ns >= 2 ||
        throw(ArgumentError("ns must be at least 2."))

    nϕ >= 1 ||
        throw(ArgumentError("nϕ must be at least 1."))

    # Get the current plane centre and in-plane basis vectors
    x0, y0, z0 = frame_position(frame)
    rx, ry, rz = frame_right(frame)
    ux, uy, uz = frame_up(frame)

    # Include both s-coordinate boundaries
    Δs = (smax - smin) / TF(ns - 1)

    # Sample the angular direction half-open, without duplicating the right boundary
    Δϕ = (ϕmax - ϕmin) / TF(nϕ)

    N = ns * nϕ

    x = Vector{TF}(undef, N)
    y = Vector{TF}(undef, N)
    z = Vector{TF}(undef, N)

    @inbounds for j in 1:nϕ
        ϕ = ϕmin + TF(j - 1) * Δϕ
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

function _cylindrical_coordinates(frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}, zparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    smin, smax, ns = sparams
    ϕmin, ϕmax, nϕ = ϕparams
    zmin, zmax, nz = zparams

    smin >= zero(TF) ||
        throw(ArgumentError("smin must be nonnegative."))

    smax > smin ||
        throw(ArgumentError("smax must be greater than smin."))

    zmax > zmin ||
        throw(ArgumentError("zmax must be greater than zmin."))

    ϕmin >= zero(TF) && (ϕmax <= TF(2π) || isapprox(ϕmax, TF(2π))) && ϕmax > ϕmin ||
        throw(ArgumentError("angular range must satisfy 0 ≤ ϕmin < ϕmax ≤ 2π."))

    ns >= 2 ||
        throw(ArgumentError("ns must be at least 2."))

    nϕ >= 1 ||
        throw(ArgumentError("nϕ must be at least 1."))

    nz >= 2 ||
        throw(ArgumentError("nz must be at least 2."))

    # Get the current centre and local cylindrical basis vectors
    x0, y0, z0 = frame_position(frame)
    rx, ry, rz = frame_right(frame)
    ux, uy, uz = frame_up(frame)
    fx, fy, fz = frame_forward(frame)

    # Include both s- and z-coordinate boundaries
    Δs = (smax - smin) / TF(ns - 1)
    Δz = (zmax - zmin) / TF(nz - 1)

    # Sample the angular direction half-open, without duplicating the right boundary
    Δϕ = (ϕmax - ϕmin) / TF(nϕ)

    N = ns * nϕ * nz

    x = Vector{TF}(undef, N)
    y = Vector{TF}(undef, N)
    z = Vector{TF}(undef, N)

    @inbounds for k3 in 1:nz
        ζ = zmin + TF(k3 - 1) * Δz

        for j in 1:nϕ
            ϕ = ϕmin + TF(j - 1) * Δϕ
            sinϕ, cosϕ = sincos(ϕ)

            @simd for i in 1:ns
                s = smin + TF(i - 1) * Δs
                k = i + (j - 1) * ns + (k3 - 1) * ns * nϕ

                ξ = s * cosϕ
                η = s * sinϕ

                x[k] = x0 + ξ * rx + η * ux + ζ * fx
                y[k] = y0 + ξ * ry + η * uy + ζ * fy
                z[k] = z0 + ξ * rz + η * uz + ζ * fz
            end
        end
    end

    return x, y, z
end

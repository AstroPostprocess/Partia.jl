######################################################################################

# Polar sample-coordinate generation.
#     by Wei-Shan Su,
#     July 16, 2026

######################################################################################

function _polar_plane_coordinates(frame :: Frame{TF}, sparams :: AxisParam{TF}, phi_params :: AxisParam{TF}) where {TF <: AbstractFloat}
    # Allocate polar-plane coordinates in structure-of-arrays form
    coords = ntuple(_ -> Vector{TF}(undef, sparams[3] * phi_params[3]), 3)
    _polar_plane_coordinates!(coords, frame, sparams, phi_params)
    return coords
end

function _polar_plane_coordinates!(coords :: NTuple{3, Vector{TF}}, frame :: Frame{TF}, sparams :: AxisParam{TF}, phi_params :: AxisParam{TF}) where {TF <: AbstractFloat}
    smin, smax, ns = sparams
    phi_min, phi_max, nphi = phi_params
    smin >= zero(TF) || throw(ArgumentError("smin must be nonnegative."))
    smax > smin || throw(ArgumentError("smax must be greater than smin."))
    phi_min >= zero(TF) && (phi_max <= TF(2π) || isapprox(phi_max, TF(2π))) && phi_max > phi_min || throw(ArgumentError("angular range must satisfy 0 ≤ phi_min < phi_max ≤ 2π."))
    ns >= 2 || throw(ArgumentError("ns must be at least 2."))
    nphi >= 1 || throw(ArgumentError("nphi must be at least 1."))
    length(coords[1]) == ns * nphi || throw(DimensionMismatch("coordinate storage must have length $(ns * nphi)"))

    # Get the current plane centre and in-plane basis vectors
    x0, y0, z0 = frame_position(frame)
    rx, ry, rz = frame_right(frame)
    ux, uy, uz = frame_up(frame)

    # Include both s-coordinate boundaries
    ds = (smax - smin) / TF(ns - 1)

    # Sample the angular direction half-open, without duplicating the right boundary
    dphi = (phi_max - phi_min) / TF(nphi)
    @inbounds for j in 1:nphi
        sin_phi, cos_phi = sincos(phi_min + TF(j - 1) * dphi)
        @simd for i in 1:ns
            s = smin + TF(i - 1) * ds
            local_x = s * cos_phi
            local_y = s * sin_phi
            index = i + (j - 1) * ns
            coords[1][index] = x0 + local_x * rx + local_y * ux
            coords[2][index] = y0 + local_x * ry + local_y * uy
            coords[3][index] = z0 + local_x * rz + local_y * uz
        end
    end
    return nothing
end

######################################################################################

# Cylindrical sample-coordinate generation.
#     by Wei-Shan Su,
#     July 16, 2026

######################################################################################

function _cylindrical_coordinates(frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}, zparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    # Allocate cylindrical coordinates in structure-of-arrays form
    coords = ntuple(_ -> Vector{TF}(undef, sparams[3] * ϕparams[3] * zparams[3]), 3)
    _cylindrical_coordinates!(coords, frame, sparams, ϕparams, zparams)
    return coords
end

function _cylindrical_coordinates!(coords :: NTuple{3, Vector{TF}}, frame :: Frame{TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}, zparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    smin, smax, ns = sparams
    ϕmin, ϕmax, nϕ = ϕparams
    zmin, zmax, nz = zparams
    smin >= zero(TF) || throw(ArgumentError("smin must be nonnegative."))
    smax > smin || throw(ArgumentError("smax must be greater than smin."))
    zmax > zmin || throw(ArgumentError("zmax must be greater than zmin."))
    ϕmin >= zero(TF) && (ϕmax <= TF(2π) || isapprox(ϕmax, TF(2π))) && ϕmax > ϕmin || throw(ArgumentError("angular range must satisfy 0 ≤ ϕmin < ϕmax ≤ 2π."))
    ns >= 2 || throw(ArgumentError("ns must be at least 2."))
    nϕ >= 1 || throw(ArgumentError("nϕ must be at least 1."))
    nz >= 2 || throw(ArgumentError("nz must be at least 2."))
    length(coords[1]) == ns * nϕ * nz || throw(DimensionMismatch("coordinate storage must have length $(ns * nϕ * nz)"))

    # Get the current centre and local cylindrical basis vectors
    x0, y0, z0 = frame_position(frame)
    rx, ry, rz = frame_right(frame)
    ux, uy, uz = frame_up(frame)
    fx, fy, fz = frame_forward(frame)

    # Include both s- and z-coordinate boundaries
    ds = (smax - smin) / TF(ns - 1)
    dz = (zmax - zmin) / TF(nz - 1)

    # Sample the angular direction half-open, without duplicating the right boundary
    Δϕ = (ϕmax - ϕmin) / TF(nϕ)
    @inbounds for k in 1:nz
        local_z = zmin + TF(k - 1) * dz
        for j in 1:nϕ
            sinϕ, cosϕ = sincos(ϕmin + TF(j - 1) * Δϕ)
            @simd for i in 1:ns
                s = smin + TF(i - 1) * ds
                local_x = s * cosϕ
                local_y = s * sinϕ
                index = i + (j - 1) * ns + (k - 1) * ns * nϕ
                coords[1][index] = x0 + local_x * rx + local_y * ux + local_z * fx
                coords[2][index] = y0 + local_x * ry + local_y * uy + local_z * fy
                coords[3][index] = z0 + local_x * rz + local_y * uz + local_z * fz
            end
        end
    end
    return nothing
end

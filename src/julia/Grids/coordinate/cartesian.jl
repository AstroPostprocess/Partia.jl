######################################################################################

# Cartesian sample-coordinate generation.
#     by Wei-Shan Su,
#     July 16, 2026

######################################################################################

function _cartesian_plane_coordinates(frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    # Allocate Cartesian coordinates in structure-of-arrays form
    coords = ntuple(_ -> Vector{TF}(undef, xparams[3] * yparams[3]), 3)
    _cartesian_plane_coordinates!(coords, frame, xparams, yparams)
    return coords
end

function _cartesian_box_coordinates(frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}, zparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    # Allocate Cartesian coordinates in structure-of-arrays form
    coords = ntuple(_ -> Vector{TF}(undef, xparams[3] * yparams[3] * zparams[3]), 3)
    _cartesian_box_coordinates!(coords, frame, xparams, yparams, zparams)
    return coords
end

function _cartesian_plane_coordinates!(coords :: NTuple{3, Vector{TF}}, frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    xmin, xmax, nx = xparams
    ymin, ymax, ny = yparams
    xmax > xmin || throw(ArgumentError("xmax must be greater than xmin."))
    ymax > ymin || throw(ArgumentError("ymax must be greater than ymin."))
    nx >= 2 || throw(ArgumentError("nx must be at least 2."))
    ny >= 2 || throw(ArgumentError("ny must be at least 2."))
    length(coords[1]) == nx * ny || throw(DimensionMismatch("coordinate storage must have length $(nx * ny)"))

    # Get the current plane centre and in-plane basis vectors
    x0, y0, z0 = frame_position(frame)
    rx, ry, rz = frame_right(frame)
    ux, uy, uz = frame_up(frame)
    # Include both Cartesian plane boundaries
    dx = (xmax - xmin) / TF(nx - 1)
    dy = (ymax - ymin) / TF(ny - 1)
    @inbounds for j in 1:ny
        local_y = ymin + TF(j - 1) * dy
        @simd for i in 1:nx
            local_x = xmin + TF(i - 1) * dx
            index = i + (j - 1) * nx
            # Map local plane coordinates to global Cartesian coordinates
            coords[1][index] = x0 + local_x * rx + local_y * ux
            coords[2][index] = y0 + local_x * ry + local_y * uy
            coords[3][index] = z0 + local_x * rz + local_y * uz
        end
    end
    return nothing
end

function _cartesian_box_coordinates!(coords :: NTuple{3, Vector{TF}}, frame :: Frame{TF}, xparams :: AxisParam{TF}, yparams :: AxisParam{TF}, zparams :: AxisParam{TF}) where {TF <: AbstractFloat}
    xmin, xmax, nx = xparams
    ymin, ymax, ny = yparams
    zmin, zmax, nz = zparams
    xmax > xmin || throw(ArgumentError("xmax must be greater than xmin."))
    ymax > ymin || throw(ArgumentError("ymax must be greater than ymin."))
    zmax > zmin || throw(ArgumentError("zmax must be greater than zmin."))
    nx >= 2 || throw(ArgumentError("nx must be at least 2."))
    ny >= 2 || throw(ArgumentError("ny must be at least 2."))
    nz >= 2 || throw(ArgumentError("nz must be at least 2."))
    length(coords[1]) == nx * ny * nz || throw(DimensionMismatch("coordinate storage must have length $(nx * ny * nz)"))

    # Get the current box centre and local basis vectors
    x0, y0, z0 = frame_position(frame)
    rx, ry, rz = frame_right(frame)
    ux, uy, uz = frame_up(frame)
    fx, fy, fz = frame_forward(frame)
    # Include all Cartesian box boundaries
    dx = (xmax - xmin) / TF(nx - 1)
    dy = (ymax - ymin) / TF(ny - 1)
    dz = (zmax - zmin) / TF(nz - 1)
    @inbounds for k in 1:nz
        local_z = zmin + TF(k - 1) * dz
        for j in 1:ny
            local_y = ymin + TF(j - 1) * dy
            @simd for i in 1:nx
                local_x = xmin + TF(i - 1) * dx
                index = i + (j - 1) * nx + (k - 1) * nx * ny
                coords[1][index] = x0 + local_x * rx + local_y * ux + local_z * fx
                coords[2][index] = y0 + local_x * ry + local_y * uy + local_z * fy
                coords[3][index] = z0 + local_x * rz + local_y * uz + local_z * fz
            end
        end
    end
    return nothing
end

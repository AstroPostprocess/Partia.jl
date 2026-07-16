@inline function _cartesian_plane_point_samples_kernel!(grid :: MtlDeviceVector{Float32, 1}, coords :: NTuple{3, MtlDeviceVector{Float32, 1}}, position :: NTuple{3, Float32}, right :: NTuple{3, Float32}, up :: NTuple{3, Float32}, xparams :: AxisParam{Float32}, yparams :: AxisParam{Float32})
    i = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)
    xmin, xmax, nx = xparams
    ymin, ymax, ny = yparams
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    while i <= length(grid)
        q = i - 1
        local_x = xmin + rem(q, nx) * dx
        local_y = ymin + div(q, nx) * dy
        @inbounds for d in 1:3
            coords[d][i] = position[d] + local_x * right[d] + local_y * up[d]
        end
        @inbounds grid[i] = zero(eltype(grid))
        i += stride
    end
    return nothing
end

@inline function _cartesian_box_point_samples_kernel!(grid :: MtlDeviceVector{Float32, 1}, coords :: NTuple{3, MtlDeviceVector{Float32, 1}}, position :: NTuple{3, Float32}, right :: NTuple{3, Float32}, up :: NTuple{3, Float32}, forward :: NTuple{3, Float32}, xparams :: AxisParam{Float32}, yparams :: AxisParam{Float32}, zparams :: AxisParam{Float32})
    i = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)
    xmin, xmax, nx = xparams
    ymin, ymax, ny = yparams
    zmin, zmax, nz = zparams
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    dz = (zmax - zmin) / (nz - 1)
    while i <= length(grid)
        q = i - 1
        ix = rem(q, nx)
        q = div(q, nx)
        iy = rem(q, ny)
        iz = div(q, ny)
        local_x = xmin + ix * dx
        local_y = ymin + iy * dy
        local_z = zmin + iz * dz
        @inbounds for d in 1:3
            coords[d][i] = position[d] + local_x * right[d] + local_y * up[d] + local_z * forward[d]
        end
        @inbounds grid[i] = zero(eltype(grid))
        i += stride
    end
    return nothing
end

@inline function _polar_point_samples_kernel!(grid :: MtlDeviceVector{Float32, 1}, coords :: NTuple{3, MtlDeviceVector{Float32, 1}}, position :: NTuple{3, Float32}, right :: NTuple{3, Float32}, up :: NTuple{3, Float32}, sparams :: AxisParam{Float32}, ϕparams :: AxisParam{Float32})
    i = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)
    smin, smax, ns = sparams
    ϕmin, ϕmax, nϕ = ϕparams
    ds = (smax - smin) / (ns - 1)
    Δϕ = (ϕmax - ϕmin) / nϕ
    while i <= length(grid)
        q = i - 1
        s = smin + rem(q, ns) * ds
        sinϕ, cosϕ = sincos(ϕmin + div(q, ns) * Δϕ)
        local_x = s * cosϕ
        local_y = s * sinϕ
        @inbounds for d in 1:3
            coords[d][i] = position[d] + local_x * right[d] + local_y * up[d]
        end
        @inbounds grid[i] = zero(eltype(grid))
        i += stride
    end
    return nothing
end

@inline function _cylindrical_point_samples_kernel!(grid :: MtlDeviceVector{Float32, 1}, coords :: NTuple{3, MtlDeviceVector{Float32, 1}}, position :: NTuple{3, Float32}, right :: NTuple{3, Float32}, up :: NTuple{3, Float32}, forward :: NTuple{3, Float32}, sparams :: AxisParam{Float32}, ϕparams :: AxisParam{Float32}, zparams :: AxisParam{Float32})
    i = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)
    smin, smax, ns = sparams
    ϕmin, ϕmax, nϕ = ϕparams
    zmin, zmax, nz = zparams
    ds = (smax - smin) / (ns - 1)
    Δϕ = (ϕmax - ϕmin) / nϕ
    dz = (zmax - zmin) / (nz - 1)
    while i <= length(grid)
        q = i - 1
        is = rem(q, ns)
        q = div(q, ns)
        iϕ = rem(q, nϕ)
        iz = div(q, nϕ)
        s = smin + is * ds
        sinϕ, cosϕ = sincos(ϕmin + iϕ * Δϕ)
        local_x = s * cosϕ
        local_y = s * sinϕ
        local_z = zmin + iz * dz
        @inbounds for d in 1:3
            coords[d][i] = position[d] + local_x * right[d] + local_y * up[d] + local_z * forward[d]
        end
        @inbounds grid[i] = zero(eltype(grid))
        i += stride
    end
    return nothing
end

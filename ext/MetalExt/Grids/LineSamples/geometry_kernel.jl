@inline function _parallel_beam_line_samples_kernel!(grid :: MtlDeviceVector{Float32, 1}, origin :: NTuple{3, MtlDeviceVector{Float32, 1}}, direction :: NTuple{3, MtlDeviceVector{Float32, 1}}, position :: NTuple{3, Float32}, right :: NTuple{3, Float32}, up :: NTuple{3, Float32}, forward :: NTuple{3, Float32}, first_params :: AxisParam{Float32}, second_params :: AxisParam{Float32}, :: Type{Cartesian})
    i = Int(Metal.thread_position_in_grid().x)
    stride = Int(Metal.threads_per_grid().x)
    xmin, xmax, nx = first_params
    ymin, ymax, ny = second_params
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    while i <= length(grid)
        q = i - 1
        local_x = xmin + rem(q, nx) * dx
        local_y = ymin + div(q, nx) * dy
        @inbounds for d in 1:3
            origin[d][i] = position[d] + local_x * right[d] + local_y * up[d]
            direction[d][i] = forward[d]
        end
        @inbounds grid[i] = zero(eltype(grid))
        i += stride
    end
    return nothing
end

@inline function _parallel_beam_line_samples_kernel!(grid :: MtlDeviceVector{Float32, 1}, origin :: NTuple{3, MtlDeviceVector{Float32, 1}}, direction :: NTuple{3, MtlDeviceVector{Float32, 1}}, position :: NTuple{3, Float32}, right :: NTuple{3, Float32}, up :: NTuple{3, Float32}, forward :: NTuple{3, Float32}, sparams :: AxisParam{Float32}, ϕparams :: AxisParam{Float32}, :: Type{Polar})
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
            origin[d][i] = position[d] + local_x * right[d] + local_y * up[d]
            direction[d][i] = forward[d]
        end
        @inbounds grid[i] = zero(eltype(grid))
        i += stride
    end
    return nothing
end

@inline function _parallel_beam_line_samples_kernel!(grid :: VG, origin :: NTuple{3, VC}, direction :: NTuple{3, VD}, position :: NTuple{3, TF}, right :: NTuple{3, TF}, up :: NTuple{3, TF}, forward :: NTuple{3, TF}, first_params :: AxisParam{TF}, second_params :: AxisParam{TF}, :: Type{Cartesian}) where {TF <: AbstractFloat, VG <: CuDeviceVector{TF}, VC <: CuDeviceVector{TF}, VD <: CuDeviceVector{TF}}
    i = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int(gridDim().x * blockDim().x)
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

@inline function _parallel_beam_line_samples_kernel!(grid :: VG, origin :: NTuple{3, VC}, direction :: NTuple{3, VD}, position :: NTuple{3, TF}, right :: NTuple{3, TF}, up :: NTuple{3, TF}, forward :: NTuple{3, TF}, sparams :: AxisParam{TF}, ϕparams :: AxisParam{TF}, :: Type{Polar}) where {TF <: AbstractFloat, VG <: CuDeviceVector{TF}, VC <: CuDeviceVector{TF}, VD <: CuDeviceVector{TF}}
    i = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int(gridDim().x * blockDim().x)
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

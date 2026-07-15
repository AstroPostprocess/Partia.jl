######################################################################################

# Structured-grid coordinate expansion driver for CUDA.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
"""
    coordinate_grid(::Type{COORD}, grid::StructuredGrid,
                    ::Val{NBlocks}=Val(256),
                    ::Val{ThreadsPerBlock}=Val(256))

Expand structured-grid axes into Cartesian structure-of-arrays coordinates on
CUDA. The output follows Julia column-major linear indexing and is compatible
with `vec(grid.grid)`.

# Parameters
- `COORD`: `Cartesian`, `Polar`, `Cylindrical`, or `Spherical`.
- `grid`: Structured grid stored in CUDA arrays.
- `NBlocks`: Number of CUDA blocks; defaults to 256.
- `ThreadsPerBlock`: Threads per CUDA block; defaults to 256.

# Returns
- An `NTuple` of CUDA vectors containing Cartesian coordinates.
"""
function Partia.coordinate_grid(:: Type{Cartesian}, grid :: StructuredGrid{D, TF, VF, AF}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {D, TF <: AbstractFloat, VF <: CuVector{TF}, AF <: CuArray{TF, D}, NBlocks, ThreadsPerBlock}
    coor = ntuple(_ -> similar(vec(grid.grid)), D)
    @cuda threads=ThreadsPerBlock blocks=NBlocks _cartesian_coordinate_grid_kernel!(coor, grid.axes, grid.size)
    CUDA.synchronize()
    return coor
end

function Partia.coordinate_grid(:: Type{Polar}, grid :: StructuredGrid{2, TF, VF, AF}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, VF <: CuVector{TF}, AF <: CuArray{TF, 2}, NBlocks, ThreadsPerBlock}
    coor = ntuple(_ -> similar(vec(grid.grid)), 2)
    @cuda threads=ThreadsPerBlock blocks=NBlocks _polar_coordinate_grid_kernel!(coor, grid.axes, grid.size)
    CUDA.synchronize()
    return coor
end

function Partia.coordinate_grid(:: Type{Cylindrical}, grid :: StructuredGrid{3, TF, VF, AF}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, VF <: CuVector{TF}, AF <: CuArray{TF, 3}, NBlocks, ThreadsPerBlock}
    coor = ntuple(_ -> similar(vec(grid.grid)), 3)
    @cuda threads=ThreadsPerBlock blocks=NBlocks _cylindrical_coordinate_grid_kernel!(coor, grid.axes, grid.size)
    CUDA.synchronize()
    return coor
end

function Partia.coordinate_grid(:: Type{Spherical}, grid :: StructuredGrid{3, TF, VF, AF}, :: Val{NBlocks} = Val(256), :: Val{ThreadsPerBlock} = Val(256)) where {TF <: AbstractFloat, VF <: CuVector{TF}, AF <: CuArray{TF, 3}, NBlocks, ThreadsPerBlock}
    coor = ntuple(_ -> similar(vec(grid.grid)), 3)
    @cuda threads=ThreadsPerBlock blocks=NBlocks _spherical_coordinate_grid_kernel!(coor, grid.axes, grid.size)
    CUDA.synchronize()
    return coor
end

######################################################################################

# Structured-grid coordinate expansion driver for Metal.
#     by Wei-Shan Su,
#     July 14, 2026

######################################################################################
"""
    coordinate_grid(::Type{COORD}, grid::StructuredGrid,
                    ::Val{NThreadgroups}=Val(128),
                    ::Val{ThreadsPerGroup}=Val(256))

Expand structured-grid axes into Cartesian structure-of-arrays coordinates on
Metal. The output follows Julia column-major linear indexing and is compatible
with `vec(grid.grid)`.
"""
function Partia.coordinate_grid(:: Type{Cartesian}, grid :: StructuredGrid{D, Float32, VF, AF}, :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {D, VF <: MtlVector{Float32}, AF <: MtlArray{Float32, D}, NThreadgroups, ThreadsPerGroup}
    coor = ntuple(_ -> similar(vec(grid.grid)), D)
    @metal threads=(ThreadsPerGroup,) groups=(NThreadgroups,) _cartesian_coordinate_grid_kernel!(coor, grid.axes, grid.size)
    Metal.synchronize()
    return coor
end

function Partia.coordinate_grid(:: Type{Polar}, grid :: StructuredGrid{2, Float32, VF, AF}, :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {VF <: MtlVector{Float32}, AF <: MtlArray{Float32, 2}, NThreadgroups, ThreadsPerGroup}
    coor = ntuple(_ -> similar(vec(grid.grid)), 2)
    @metal threads=(ThreadsPerGroup,) groups=(NThreadgroups,) _polar_coordinate_grid_kernel!(coor, grid.axes, grid.size)
    Metal.synchronize()
    return coor
end

function Partia.coordinate_grid(:: Type{Cylindrical}, grid :: StructuredGrid{3, Float32, VF, AF}, :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {VF <: MtlVector{Float32}, AF <: MtlArray{Float32, 3}, NThreadgroups, ThreadsPerGroup}
    coor = ntuple(_ -> similar(vec(grid.grid)), 3)
    @metal threads=(ThreadsPerGroup,) groups=(NThreadgroups,) _cylindrical_coordinate_grid_kernel!(coor, grid.axes, grid.size)
    Metal.synchronize()
    return coor
end

function Partia.coordinate_grid(:: Type{Spherical}, grid :: StructuredGrid{3, Float32, VF, AF}, :: Val{NThreadgroups} = Val(128), :: Val{ThreadsPerGroup} = Val(256)) where {VF <: MtlVector{Float32}, AF <: MtlArray{Float32, 3}, NThreadgroups, ThreadsPerGroup}
    coor = ntuple(_ -> similar(vec(grid.grid)), 3)
    @metal threads=(ThreadsPerGroup,) groups=(NThreadgroups,) _spherical_coordinate_grid_kernel!(coor, grid.axes, grid.size)
    Metal.synchronize()
    return coor
end

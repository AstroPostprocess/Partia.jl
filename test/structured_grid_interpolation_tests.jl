######################################################################################

#  Test: StructuredGrid Interpolation and Transformations
#  What this file tests
#  Validates structured-grid transformations and interpolation:
#  1. StructuredGrid transformation
#     • Flatten and restore round-trip for Cartesian grids.
#     • Coordinate dispatch for Cartesian, polar, cylindrical, and spherical
#       grids against explicitly constructed Cartesian coordinates.
#     • Coordinate-dispatch mismatch rejection during restore.
#  2. Frame-plane sample constructors
#     • Cartesian and polar constructors checked against equivalent 2D
#       StructuredGrid flattening.
#     • Independent half-open polar angular sampling and radial ordering checks.
#     • LineSamples parallel-beam constructors checked against PointSamples
#       origins and frame-forward directions.
#  3. StructuredGrid interpolation
#     • Flattened PointSamples interpolation equivalence.
#     • Coordinate-system dispatch and analytic linear-field regression.

######################################################################################
using Test
using Random
using Partia

# ========================== Shared includes ================================= #

@static if !isdefined(@__MODULE__, :make_grid_interpolation_fixture)
    include("grid_interpolation_test_common.jl")
end

# ============================== Test body =================================== #

# ── 1a. StructuredGrid transform — flatten and restore ───────────────── #

@testset "StructuredGrid transform -- flatten and restore" begin
    grid = make_structured_grid_template()
    grid.grid .= reshape(collect(1.0:length(grid.grid)), size(grid))

    flattened = Partia.Grids.flatten(Cartesian, grid)
    restored = Partia.Grids.restore_struct(Cartesian, flattened, grid.axes)

    @test restored.grid == grid.grid
    @test restored.axes == grid.axes
    @test restored.size == size(grid)
end

# ── 1b. StructuredGrid transform — coordinate dispatch ───────────────── #

@testset "StructuredGrid transform -- coordinate dispatch" begin
    for (coord, grid) in (
        (Cartesian, make_structured_grid_template()),
        (Polar, make_polar_grid_template()),
        (Cylindrical, make_cylindrical_grid_template()),
        (Spherical, make_spherical_grid_template()),
    )
        grid.grid .= reshape(collect(1.0:length(grid.grid)), size(grid))
        flattened = Partia.Grids.flatten(coord, grid)
        expected = explicit_cartesian_coords(coord, grid)

        @test flattened.grid == vec(grid.grid)
        @test length(flattened.coor) == length(expected)
        @test all(isapprox.(flattened.coor[1], expected[1]; atol = 1.0e-12, rtol = 1.0e-12))
        @test all(isapprox.(flattened.coor[2], expected[2]; atol = 1.0e-12, rtol = 1.0e-12))
        if length(expected) == 3
            @test all(isapprox.(flattened.coor[3], expected[3]; atol = 1.0e-12, rtol = 1.0e-12))
        end

        restored = Partia.Grids.restore_struct(coord, flattened, grid.axes)
        @test restored.grid == grid.grid
        @test restored.axes == grid.axes
        @test restored.size == size(grid)
    end
end

# ── 1c. StructuredGrid restore — dispatch mismatch ───────────────────── #

@testset "StructuredGrid restore -- coordinate dispatch mismatch is rejected" begin
    cyl_grid = make_cylindrical_grid_template()
    flattened = Partia.Grids.flatten(Cylindrical, cyl_grid)

    @test_throws ArgumentError Partia.Grids.restore_struct(Cartesian, flattened, cyl_grid.axes)
end

# ── 2a. PointSamples frame-plane constructors ────────────────────────── #

@testset "PointSamples frame-plane constructors -- 2D StructuredGrid consistency" begin
    frame = Frame((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))

    xparams = (-1.0, 1.0, 5)
    yparams = (-0.75, 0.75, 4)
    structured_cart = StructuredGrid(Cartesian, xparams, yparams)
    flattened_cart = Partia.Grids.flatten(Cartesian, structured_cart)
    plane_cart = PointSamples(Cartesian, frame, xparams, yparams)

    @test plane_cart.grid == zeros(length(flattened_cart.grid))
    @test isapprox(plane_cart.coor[1], flattened_cart.coor[1]; atol = 1.0e-12, rtol = 1.0e-12)
    @test isapprox(plane_cart.coor[2], flattened_cart.coor[2]; atol = 1.0e-12, rtol = 1.0e-12)
    @test all(isapprox.(plane_cart.coor[3], 0.0; atol = 1.0e-12, rtol = 1.0e-12))

    zparams = (-0.5, 0.5, 3)
    box_cart = PointSamples(Cartesian, frame, xparams, yparams, zparams)

    @test length(box_cart.grid) == xparams[3] * yparams[3] * zparams[3]
    @test box_cart.grid == zeros(length(box_cart.grid))
    @test isapprox(box_cart.coor[1][1], xparams[1]; atol = 1.0e-12, rtol = 1.0e-12)
    @test isapprox(box_cart.coor[2][1], yparams[1]; atol = 1.0e-12, rtol = 1.0e-12)
    @test isapprox(box_cart.coor[3][1], -zparams[1]; atol = 1.0e-12, rtol = 1.0e-12)
    @test isapprox(box_cart.coor[1][end], xparams[2]; atol = 1.0e-12, rtol = 1.0e-12)
    @test isapprox(box_cart.coor[2][end], yparams[2]; atol = 1.0e-12, rtol = 1.0e-12)
    @test isapprox(box_cart.coor[3][end], -zparams[2]; atol = 1.0e-12, rtol = 1.0e-12)

    smin, smax = 0.0, 1.0
    ns, nϕ = 4, 8
    sparams = (smin, smax, ns)
    ϕparams = (0.0, 2π, nϕ)
    structured_polar = StructuredGrid(Polar, sparams, ϕparams)
    flattened_polar = Partia.Grids.flatten(Polar, structured_polar)
    plane_polar = PointSamples(Polar, frame, sparams, ϕparams)

    @test plane_polar.grid == zeros(length(flattened_polar.grid))
    @test isapprox(plane_polar.coor[1], flattened_polar.coor[1]; atol = 1.0e-12, rtol = 1.0e-12)
    @test isapprox(plane_polar.coor[2], flattened_polar.coor[2]; atol = 1.0e-12, rtol = 1.0e-12)
    @test all(isapprox.(plane_polar.coor[3], 0.0; atol = 1.0e-12, rtol = 1.0e-12))

    partial_ϕparams = (π / 4, 3π / 4, 5)
    partial_structured_polar = StructuredGrid(Polar, sparams, partial_ϕparams)
    partial_flattened_polar = Partia.Grids.flatten(Polar, partial_structured_polar)
    partial_plane_polar = PointSamples(Polar, frame, sparams, partial_ϕparams)

    @test isapprox(partial_plane_polar.coor[1], partial_flattened_polar.coor[1]; atol = 1.0e-12, rtol = 1.0e-12)
    @test isapprox(partial_plane_polar.coor[2], partial_flattened_polar.coor[2]; atol = 1.0e-12, rtol = 1.0e-12)
    @test all(isapprox.(partial_plane_polar.coor[3], 0.0; atol = 1.0e-12, rtol = 1.0e-12))

    scoords = range(smin, smax; length = ns)
    angles = range(0.0, 2π; length = nϕ + 1)[1:end-1]

    expected_x = Vector{Float64}(undef, ns * nϕ)
    expected_y = Vector{Float64}(undef, ns * nϕ)
    expected_z = zeros(Float64, ns * nϕ)

    @inbounds for j in 1:nϕ
        ϕ = angles[j]
        for i in 1:ns
            k = i + (j - 1) * ns
            s = scoords[i]

            expected_x[k] = s * cos(ϕ)
            expected_y[k] = s * sin(ϕ)
        end
    end

    @test isapprox(plane_polar.coor[1], expected_x; atol = 1.0e-12, rtol = 1.0e-12)
    @test isapprox(plane_polar.coor[2], expected_y; atol = 1.0e-12, rtol = 1.0e-12)
    @test isapprox(plane_polar.coor[3], expected_z; atol = 1.0e-12, rtol = 1.0e-12)

    @test length(plane_polar.grid) == ns * nϕ
    @test plane_polar.grid == zeros(ns * nϕ)

    @test isapprox(plane_polar.coor[1][1:ns], collect(scoords); atol = 1.0e-12, rtol = 1.0e-12)
    @test all(isapprox.(plane_polar.coor[2][1:ns], 0.0; atol = 1.0e-12, rtol = 1.0e-12))

    outer_indices = ns:ns:(ns * nϕ)
    @test all(isapprox.(
        hypot.(plane_polar.coor[1][outer_indices], plane_polar.coor[2][outer_indices]),
        smax;
        atol = 1.0e-12,
        rtol = 1.0e-12,
    ))

    centre_indices = 1:ns:(ns * nϕ)
    @test all(isapprox.(plane_polar.coor[1][centre_indices], 0.0; atol = 1.0e-12, rtol = 1.0e-12))
    @test all(isapprox.(plane_polar.coor[2][centre_indices], 0.0; atol = 1.0e-12, rtol = 1.0e-12))
    @test all(isapprox.(plane_polar.coor[3][centre_indices], 0.0; atol = 1.0e-12, rtol = 1.0e-12))
end

# ── 2b. LineSamples frame-plane constructors ─────────────────────────── #

@testset "LineSamples frame-plane constructors -- ParallelBeam consistency" begin
    frame = Frame((0.25, -0.5, 1.25), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    fx, fy, fz = frame_forward(frame)

    xparams = (-1.0, 1.0, 5)
    yparams = (-0.75, 0.75, 4)
    point_cart = PointSamples(Cartesian, frame, xparams, yparams)
    line_cart = LineSamples(Cartesian, ParallelBeam, frame, xparams, yparams)

    @test line_cart.grid == zeros(length(point_cart.grid))
    @test isapprox(line_cart.origin[1], point_cart.coor[1]; atol = 1.0e-12, rtol = 1.0e-12)
    @test isapprox(line_cart.origin[2], point_cart.coor[2]; atol = 1.0e-12, rtol = 1.0e-12)
    @test isapprox(line_cart.origin[3], point_cart.coor[3]; atol = 1.0e-12, rtol = 1.0e-12)
    @test all(isapprox.(line_cart.direction[1], fx; atol = 1.0e-12, rtol = 1.0e-12))
    @test all(isapprox.(line_cart.direction[2], fy; atol = 1.0e-12, rtol = 1.0e-12))
    @test all(isapprox.(line_cart.direction[3], fz; atol = 1.0e-12, rtol = 1.0e-12))

    smin, smax = 0.0, 1.0
    ns, nϕ = 4, 8
    sparams = (smin, smax, ns)
    ϕparams = (0.0, 2π, nϕ)
    point_polar = PointSamples(Polar, frame, sparams, ϕparams)
    line_polar = LineSamples(Polar, ParallelBeam, frame, sparams, ϕparams)

    @test line_polar.grid == zeros(length(point_polar.grid))
    @test isapprox(line_polar.origin[1], point_polar.coor[1]; atol = 1.0e-12, rtol = 1.0e-12)
    @test isapprox(line_polar.origin[2], point_polar.coor[2]; atol = 1.0e-12, rtol = 1.0e-12)
    @test isapprox(line_polar.origin[3], point_polar.coor[3]; atol = 1.0e-12, rtol = 1.0e-12)
    @test all(isapprox.(line_polar.direction[1], fx; atol = 1.0e-12, rtol = 1.0e-12))
    @test all(isapprox.(line_polar.direction[2], fy; atol = 1.0e-12, rtol = 1.0e-12))
    @test all(isapprox.(line_polar.direction[3], fz; atol = 1.0e-12, rtol = 1.0e-12))

    @test length(line_polar.grid) == ns * nϕ
    @test length(line_polar.origin[1]) == ns * nϕ
    @test length(line_polar.direction[1]) == ns * nϕ
end

@testset "LineSamples frame-plane constructors -- invalid plane parameters" begin
    frame = Frame((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))

    @test_throws ArgumentError LineSamples(Cartesian, ParallelBeam, frame, (0.0, 0.0, 2), (0.0, 1.0, 2))
    @test_throws ArgumentError LineSamples(Cartesian, ParallelBeam, frame, (0.0, 1.0, 1), (0.0, 1.0, 2))
    @test_throws ArgumentError LineSamples(Cartesian, ParallelBeam, frame, (0.0, 1.0, 2), (0.0, 0.0, 2))
    @test_throws ArgumentError LineSamples(Cartesian, ParallelBeam, frame, (0.0, 1.0, 2), (0.0, 1.0, 1))

    @test_throws ArgumentError LineSamples(Polar, ParallelBeam, frame, (-1.0, 1.0, 2), (0.0, 2π, 3))
    @test_throws ArgumentError LineSamples(Polar, ParallelBeam, frame, (1.0, 1.0, 2), (0.0, 2π, 3))
    @test_throws ArgumentError LineSamples(Polar, ParallelBeam, frame, (0.0, 1.0, 1), (0.0, 2π, 3))
    @test_throws ArgumentError LineSamples(Polar, ParallelBeam, frame, (0.0, 1.0, 2), (0.0, 2π, 0))
    @test_throws ArgumentError LineSamples(Polar, ParallelBeam, frame, (0.0, 1.0, 2), (Float64(pi), 0.0, 3))
end

# ── 3a. StructuredGrid interpolation — flattened consistency ─────────── #

@testset "StructuredGrid interpolation -- flattened consistency" begin
    input, catalog, _ = make_grid_interpolation_fixture()
    structured_template = make_structured_grid_template()
    flattened_template = Partia.Grids.flatten(Cartesian, structured_template)

    point_result = PointSamples_interpolation(
        CPUComputeBackend(),
        flattened_template,
        input,
        catalog,
        itpSymmetric,
    )
    structured_result = StructuredGrid_interpolation(
        CPUComputeBackend(),
        Cartesian,
        structured_template,
        input,
        catalog,
        itpSymmetric,
    )

    @test structured_result.names == point_result.names
    @test structured_result.names == catalog.ordered_names

    for i in eachindex(structured_result.grids)
        @test structured_result.grids[i].axes == structured_template.axes
        @test isapprox(vec(structured_result.grids[i].grid), point_result.grids[i].grid; atol = 1.0e-12, rtol = 1.0e-10)
    end
end

# ── 3b. StructuredGrid interpolation — coordinate-system dispatch ────── #

@testset "StructuredGrid interpolation -- flattened consistency by coordinate system" begin
    input, catalog, _ = make_grid_interpolation_fixture()

    for (coord, template) in (
        (Cartesian, make_structured_grid_template()),
        (Cylindrical, make_cylindrical_grid_template()),
        (Spherical, make_spherical_grid_template()),
    )
        flattened_template = Partia.Grids.flatten(coord, template)
        point_result = PointSamples_interpolation(
            CPUComputeBackend(),
            flattened_template,
            input,
            catalog,
            itpSymmetric,
        )
        structured_result = StructuredGrid_interpolation(
            CPUComputeBackend(),
            coord,
            template,
            input,
            catalog,
            itpSymmetric,
        )

        @test structured_result.names == point_result.names
        @test structured_result.names == catalog.ordered_names

        for i in eachindex(structured_result.grids)
            @test structured_result.grids[i].axes == template.axes
            @test isapprox(vec(structured_result.grids[i].grid), point_result.grids[i].grid; atol = 1.0e-12, rtol = 1.0e-10)
        end
    end
end

# ── 3c. StructuredGrid interpolation — analytic regression ───────────── #

@testset "StructuredGrid interpolation -- analytic linear-field regression" begin
    input, catalog, _ = make_uniform_cloud_3d(12; eta = 1.2, variable_h = true)
    for (coord_name, coord, structured_template) in (
        ("Cartesian", Cartesian, make_analytic_structured_grid()),
        ("Cylindrical", Cylindrical, make_analytic_cylindrical_grid()),
        ("Spherical", Spherical, make_analytic_spherical_grid()),
    )
        @testset "$coord_name grid" begin
            sample_coords = explicit_cartesian_coords(coord, structured_template)

            for strategy in (itpGather, itpScatter, itpSymmetric)
                result = StructuredGrid_interpolation(CPUComputeBackend(), coord, structured_template, input, catalog, strategy)

                q_grid = vec(result.grids[1].grid)
                gradx_grid = vec(result.grids[2].grid)
                grady_grid = vec(result.grids[3].grid)
                gradz_grid = vec(result.grids[4].grid)
                div_grid = vec(result.grids[5].grid)
                curlx_grid = vec(result.grids[6].grid)
                curly_grid = vec(result.grids[7].grid)
                curlz_grid = vec(result.grids[8].grid)

                for i in eachindex(q_grid)
                    point = (
                        sample_coords[1][i],
                        sample_coords[2][i],
                        sample_coords[3][i],
                    )
                    s_ref = analytic_scalar(point...)
                    g_ref = analytic_grad_scalar(point...)
                    div_ref = analytic_divA(point...)
                    curl_ref = analytic_curlA(point...)

                    @test isapprox(q_grid[i], s_ref; atol = 2e-2, rtol = 1e-2)
                    @test isapprox(gradx_grid[i], g_ref[1]; atol = 6e-2, rtol = 1e-2)
                    @test isapprox(grady_grid[i], g_ref[2]; atol = 6e-2, rtol = 1e-2)
                    @test isapprox(gradz_grid[i], g_ref[3]; atol = 6e-2, rtol = 1e-2)
                    @test isapprox(div_grid[i], div_ref; atol = 1e-1, rtol = 2e-2)
                    @test isapprox(curlx_grid[i], curl_ref[1]; atol = 5e-2, rtol = 1e-2)
                    @test isapprox(curly_grid[i], curl_ref[2]; atol = 5e-2, rtol = 1e-2)
                    @test isapprox(curlz_grid[i], curl_ref[3]; atol = 5e-2, rtol = 1e-2)
                end
            end
        end
    end
end

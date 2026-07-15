######################################################################################

#  Shared Test Helpers: CUDA and Metal Regression Tests
#  What this file provides
#  A backend-neutral regression suite for the optional GPU extensions:
#  1. Host/device data movement
#     • Interpolation inputs, sample grids, structured grids, Morton encodings,
#       and LinearBVH objects round-trip through the extension helpers.
#  2. Spatial data structures
#     • GPU Morton encoding and LinearBVH construction are compared with the
#       equivalent Float32 CPU results for 2D and 3D inputs.
#     • Identical Morton codes and finite leaf AABBs are covered explicitly.
#  3. Interpolation
#     • PointSamples Gather and Scatter interpolation.
#     • InterpolationInput and InterpolationSmoothingVolumeInput paths.
#     • LineSamples scatter interpolation.
#     • StructuredGrid interpolation for Cartesian, cylindrical, and spherical
#       coordinate dispatch.
#  The same test body is called for CUDA and Metal so both extensions are held
#  to the same functional contract.

######################################################################################
using Test
using Random
using Partia
using UnsignedRadixSorts

# ========================== Shared includes ================================= #

@static if !isdefined(@__MODULE__, :make_grid_interpolation_fixture)
    include("grid_interpolation_test_common.jl")
end

# ========================== Float32 test copies ============================== #

function accelerator_test_float32(input :: InterpolationInput{D, T, V, K, N}) where {D, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel, N}
    return InterpolationInput(
        ntuple(d -> Float32.(input.coord[d]), D),
        Float32.(input.m),
        Float32.(input.h),
        Float32.(input.ρ),
        ntuple(i -> Float32.(input.quant[i]), N);
        smoothed_kernel = K,
    )
end

function accelerator_test_float32(input :: InterpolationSmoothingVolumeInput{D, T, V, K, N}) where {D, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel, N}
    return InterpolationSmoothingVolumeInput(
        Float32(input.hfact),
        ntuple(d -> Float32.(input.coord[d]), D),
        Float32.(input.m),
        Float32.(input.h),
        ntuple(i -> Float32.(input.quant[i]), N);
        smoothed_kernel = K,
    )
end

function accelerator_test_float32(grid :: PointSamples{D, T}) where {D, T <: AbstractFloat}
    return PointSamples(
        Float32.(grid.grid),
        ntuple(d -> Float32.(grid.coor[d]), D),
    )
end

function accelerator_test_float32(grid :: LineSamples{D, T}) where {D, T <: AbstractFloat}
    return LineSamples(
        Float32.(grid.grid),
        ntuple(d -> Float32.(grid.origin[d]), D),
        ntuple(d -> Float32.(grid.direction[d]), D),
    )
end

function accelerator_test_float32(grid :: StructuredGrid{D, T}) where {D, T <: AbstractFloat}
    return StructuredGrid(
        zeros(Float32, grid.size),
        ntuple(d -> Float32.(grid.axes[d]), D),
        grid.size,
    )
end

# ========================== Comparison helpers ============================== #

function accelerator_test_host_bundle(bundle :: GridBundle, to_host)
    return GridBundle(map(to_host, bundle.grids), bundle.names)
end

function accelerator_test_grid_equal(actual :: PointSamples, expected :: PointSamples; atol, rtol)
    @test length(actual) == length(expected)
    @test length(actual.coor) == length(expected.coor)
    @test approx_with_nan(actual.grid, expected.grid; atol, rtol)
    @test approx_with_nan(actual.coor, expected.coor; atol, rtol)
    return nothing
end

function accelerator_test_grid_equal(actual :: LineSamples, expected :: LineSamples; atol, rtol)
    @test length(actual) == length(expected)
    @test length(actual.origin) == length(expected.origin)
    @test approx_with_nan(actual.grid, expected.grid; atol, rtol)
    @test approx_with_nan(actual.origin, expected.origin; atol, rtol)
    @test approx_with_nan(actual.direction, expected.direction; atol, rtol)
    return nothing
end

function accelerator_test_grid_equal(actual :: StructuredGrid, expected :: StructuredGrid; atol, rtol)
    @test actual.size == expected.size
    @test approx_with_nan(actual.grid, expected.grid; atol, rtol)
    @test approx_with_nan(actual.axes, expected.axes; atol, rtol)
    return nothing
end

function accelerator_test_bundle_equal(actual :: GridBundle, expected :: GridBundle; atol, rtol)
    @test actual.names == expected.names
    @test length(actual.grids) == length(expected.grids)
    @inbounds for i in eachindex(actual.grids, expected.grids)
        accelerator_test_grid_equal(actual.grids[i], expected.grids[i]; atol, rtol)
    end
    return nothing
end

function accelerator_test_input_equal(actual :: InterpolationInput, expected :: InterpolationInput; atol, rtol)
    @test actual.Npart == expected.Npart
    @test typeof(actual.smoothed_kernel) === typeof(expected.smoothed_kernel)
    @test approx_with_nan(actual.coord, expected.coord; atol, rtol)
    @test approx_with_nan(actual.m, expected.m; atol, rtol)
    @test approx_with_nan(actual.h, expected.h; atol, rtol)
    @test approx_with_nan(actual.ρ, expected.ρ; atol, rtol)
    @test approx_with_nan(actual.quant, expected.quant; atol, rtol)
    return nothing
end

function accelerator_test_input_equal(actual :: InterpolationSmoothingVolumeInput, expected :: InterpolationSmoothingVolumeInput; atol, rtol)
    @test actual.Npart == expected.Npart
    @test typeof(actual.smoothed_kernel) === typeof(expected.smoothed_kernel)
    @test isapprox(actual.hfact, expected.hfact; atol, rtol)
    @test approx_with_nan(actual.coord, expected.coord; atol, rtol)
    @test approx_with_nan(actual.m, expected.m; atol, rtol)
    @test approx_with_nan(actual.h, expected.h; atol, rtol)
    @test approx_with_nan(actual.quant, expected.quant; atol, rtol)
    return nothing
end

function accelerator_test_encoding_equal(actual :: MortonEncoding, expected :: MortonEncoding; atol, rtol)
    @test actual.order == expected.order
    @test actual.codes == expected.codes
    @test approx_with_nan(actual.coord, expected.coord; atol, rtol)
    return nothing
end

function accelerator_test_lbvh_equal(actual :: LinearBVH, expected :: LinearBVH; atol, rtol)
    @test actual.nleaf == expected.nleaf
    @test actual.left == expected.left
    @test actual.escape == expected.escape
    @test approx_with_nan(actual.aabb.min, expected.aabb.min; atol, rtol)
    @test approx_with_nan(actual.aabb.max, expected.aabb.max; atol, rtol)
    @test approx_with_nan(actual.scale, expected.scale; atol, rtol)
    return nothing
end

@inline accelerator_test_right_child(lbvh, node) = lbvh.escape[Int(lbvh.left[Int(node)])]

function accelerator_test_visit_nodes(lbvh)
    visited = Int[]
    node = Int32(1)
    while !iszero(node)
        push!(visited, Int(node))
        node = Partia.LinearBoundingVolumeHierarchy.is_leaf_id(node, lbvh.nleaf) ?
            lbvh.escape[Int(node)] : lbvh.left[Int(node)]
    end
    return visited
end

# ========================== Fixture helpers ================================= #

function accelerator_test_morton_coordinates(::Val{D}, n, seed) where {D}
    rng = MersenneTwister(seed)
    base = Float32.(0:n-1) ./ Float32(n - 1)
    return ntuple(_ -> base[randperm(rng, n)], D)
end

function accelerator_test_grid_fixture()
    input, catalog, _ = make_grid_interpolation_fixture()
    return accelerator_test_float32(input), catalog
end

function accelerator_test_line_fixture()
    input, catalog, _ = make_line_interpolation_fixture()
    return accelerator_test_float32(input), catalog
end

# ========================== Shared backend suite ============================ #

function run_accelerator_test_suite(config)
    name = config.name
    to_device = config.to_device
    to_device_vector = config.to_device_vector
    to_host = config.to_host
    synchronize = config.synchronize

    atol = 5.0f-5
    rtol = 5.0f-4

    # ── 1. Host/device movement ────────────────────────────────────────── #

    @testset "$name -- data movement" begin
        input, _ = accelerator_test_grid_fixture()
        device_input = to_device(input)
        synchronize()
        accelerator_test_input_equal(to_host(device_input), input; atol, rtol)

        standard_input, smoothing_input, _ = make_smoothing_volume_grid_interpolation_fixture()
        smoothing_input32 = accelerator_test_float32(smoothing_input)
        device_smoothing_input = to_device(smoothing_input32)
        synchronize()
        accelerator_test_input_equal(to_host(device_smoothing_input), smoothing_input32; atol, rtol)

        point_template = accelerator_test_float32(make_point_samples_template())
        line_template = accelerator_test_float32(make_line_samples_template())
        structured_template = accelerator_test_float32(make_structured_grid_template())

        accelerator_test_grid_equal(to_host(to_device(point_template)), point_template; atol, rtol)
        accelerator_test_grid_equal(to_host(to_device(line_template)), line_template; atol, rtol)
        accelerator_test_grid_equal(to_host(to_device(structured_template)), structured_template; atol, rtol)

        device_point = PointSamples(
            to_device_vector(point_template.coor[1]),
            to_device_vector(point_template.coor[2]),
            to_device_vector(point_template.coor[3]),
        )
        device_line = LineSamples(
            to_device_vector(line_template.origin[1]),
            to_device_vector(line_template.origin[2]),
            to_device_vector(line_template.origin[3]),
            to_device_vector(line_template.direction[1]),
            to_device_vector(line_template.direction[2]),
            to_device_vector(line_template.direction[3]),
        )
        line2_template = LineSamples(
            zeros(Float32, length(line_template)),
            (line_template.origin[1], line_template.origin[2]),
            (line_template.direction[1], line_template.direction[2]),
        )
        device_line2 = LineSamples(
            to_device_vector(line2_template.origin[1]),
            to_device_vector(line2_template.origin[2]),
            to_device_vector(line2_template.direction[1]),
            to_device_vector(line2_template.direction[2]),
        )
        synchronize()
        accelerator_test_grid_equal(to_host(device_point), point_template; atol, rtol)
        accelerator_test_grid_equal(to_host(device_line), line_template; atol, rtol)
        accelerator_test_grid_equal(to_host(device_line2), line2_template; atol, rtol)

    end

    # ── 2. Morton encoding and LinearBVH ───────────────────────────────── #

    @testset "$name -- MortonEncoding and LinearBVH" begin
        # Large bounds reductions must remain scalar-valued. This specifically
        # guards Metal against the tuple-valued `extrema` reduction that can
        # reset its command buffer at this problem size.
        large_bounds_coords = (
            collect(range(-2.0f0, 3.0f0; length = 100_000)),
            collect(range(4.0f0, -1.0f0; length = 100_000)),
        )
        device_bounds_coords = map(to_device_vector, large_bounds_coords)
        device_bounds = Partia.LinearBoundingVolumeHierarchy._coordinate_bounds(device_bounds_coords)
        synchronize()
        @test device_bounds == ((-2.0f0, 3.0f0), (-1.0f0, 4.0f0))

        for D in (Val(2), Val(3))
            dim = D isa Val{2} ? 2 : 3
            coords = accelerator_test_morton_coordinates(D, 257, 0xBEEF + dim)
            scale = collect(range(0.01f0, 0.20f0; length = length(coords[1])))

            cpu_enc = MortonEncoding(coords)
            gpu_points = ntuple(d -> to_device_vector(coords[d]), dim)
            gpu_enc = MortonEncoding(gpu_points)
            synchronize()
            host_enc = to_host(gpu_enc)
            accelerator_test_encoding_equal(host_enc, cpu_enc; atol, rtol)

            # Rebuild the same encoding and sorting workspace through the
            # coordinate-wise public build! API.
            gpu_workspace = OnesweepWorkspace(typeof(gpu_enc.codes))
            build_result = dim == 2 ?
                build!(gpu_enc, gpu_points[1], gpu_points[2], gpu_workspace) :
                build!(gpu_enc, gpu_points[1], gpu_points[2], gpu_points[3], gpu_workspace)
            @test isnothing(build_result)
            synchronize()
            accelerator_test_encoding_equal(to_host(gpu_enc), cpu_enc; atol, rtol)

            no_copy_points = ntuple(d -> to_device_vector(coords[d]), dim)
            no_copy_enc = dim == 2 ?
                MortonEncoding!(no_copy_points[1], no_copy_points[2]) :
                MortonEncoding!(no_copy_points[1], no_copy_points[2], no_copy_points[3])
            @test no_copy_enc.coord === no_copy_points
            synchronize()
            accelerator_test_encoding_equal(to_host(no_copy_enc), cpu_enc; atol, rtol)

            adapted_enc = to_device(cpu_enc)
            synchronize()
            accelerator_test_encoding_equal(to_host(adapted_enc), cpu_enc; atol, rtol)

            sorted_scale = scale[Int.(cpu_enc.order)]
            cpu_lbvh = LinearBVH(cpu_enc, sorted_scale)
            gpu_lbvh = LinearBVH(gpu_enc, to_device_vector(sorted_scale))
            synchronize()
            host_lbvh = to_host(gpu_lbvh)
            accelerator_test_lbvh_equal(host_lbvh, cpu_lbvh; atol, rtol)
            @test sort(accelerator_test_visit_nodes(host_lbvh)) == collect(1:(2 * host_lbvh.nleaf - 1))

            # Rebuild the same hierarchy and rendezvous storage in place.
            gpu_store = to_device_vector(fill(Int32(7), length(sorted_scale) - 1))
            build_result = build!(gpu_lbvh, gpu_store, gpu_enc, to_device_vector(sorted_scale))
            @test isnothing(build_result)
            synchronize()
            accelerator_test_lbvh_equal(to_host(gpu_lbvh), cpu_lbvh; atol, rtol)

            adapted_lbvh = to_device(cpu_lbvh)
            synchronize()
            accelerator_test_lbvh_equal(to_host(adapted_lbvh), cpu_lbvh; atol, rtol)
        end
    end

    @testset "$name -- LinearBVH -- identical Morton codes" begin
        for D in (2, 3)
            coords = ntuple(_ -> fill(0.5f0, 17), D)
            gpu_enc = MortonEncoding(ntuple(d -> to_device_vector(coords[d]), D))
            gpu_lbvh = LinearBVH(gpu_enc, to_device_vector(ones(Float32, 17)))
            synchronize()
            host_lbvh = to_host(gpu_lbvh)

            @test sort(accelerator_test_visit_nodes(host_lbvh)) == collect(1:33)
            @test all(==(0.5f0), (host_lbvh.aabb.min[d][1] for d in 1:D))
            @test all(==(0.5f0), (host_lbvh.aabb.max[d][1] for d in 1:D))
        end
    end

    @testset "$name -- LinearBVH -- finite leaf AABB" begin
        coords = (Float32[0.5], Float32[1.5])
        leaf_min = (Float32[0.0], Float32[1.0])
        leaf_max = (Float32[1.0], Float32[2.0])
        gpu_enc = MortonEncoding(map(to_device_vector, coords))
        gpu_lbvh = LinearBVH(
            gpu_enc,
            to_device_vector(Float32[0.1]),
            map(to_device_vector, leaf_min),
            map(to_device_vector, leaf_max),
        )
        synchronize()
        host_lbvh = to_host(gpu_lbvh)

        @test host_lbvh.aabb.min == leaf_min
        @test host_lbvh.aabb.max == leaf_max
        @test host_lbvh.scale == Float32[0.1]
        @test host_lbvh.escape == Int32[0]
    end

    # ── 3. PointSamples interpolation ──────────────────────────────────── #

    @testset "$name -- PointSamples interpolation -- CPU consistency" begin
        point_template = accelerator_test_float32(make_point_samples_template())

        for strategy in (itpGather, itpScatter)
            source_input, catalog = accelerator_test_grid_fixture()
            cpu_result = PointSamples_interpolation(point_template, deepcopy(source_input), catalog, strategy)

            gpu_input = to_device(source_input)
            gpu_template = to_device(point_template)
            gpu_result = PointSamples_interpolation(gpu_template, gpu_input, catalog, strategy)
            @test same_coordinates(gpu_result.grids...)
            synchronize()

            host_result = accelerator_test_host_bundle(gpu_result, to_host)
            accelerator_test_bundle_equal(host_result, cpu_result; atol, rtol)
        end
    end

    @testset "$name -- PointSamples interpolation -- externally supplied LBVH" begin
        source_input, catalog = accelerator_test_grid_fixture()
        point_template = accelerator_test_float32(make_point_samples_template())

        auto_input = to_device(source_input)
        auto_result = PointSamples_interpolation(to_device(point_template), auto_input, catalog, itpScatter)

        manual_input = to_device(source_input)
        manual_lbvh = LinearBVH!(manual_input)
        manual_result = PointSamples_interpolation(
            to_device(point_template),
            manual_input,
            catalog,
            manual_lbvh,
            itpScatter,
        )
        @test same_coordinates(auto_result.grids...)
        @test same_coordinates(manual_result.grids...)
        synchronize()

        host_auto = accelerator_test_host_bundle(auto_result, to_host)
        host_manual = accelerator_test_host_bundle(manual_result, to_host)
        accelerator_test_bundle_equal(host_manual, host_auto; atol, rtol)
    end

    @testset "$name -- PointSamples interpolation -- smoothing-volume consistency" begin
        standard_input, smoothing_input, catalog = make_smoothing_volume_grid_interpolation_fixture()
        standard_input32 = accelerator_test_float32(standard_input)
        smoothing_input32 = accelerator_test_float32(smoothing_input)
        point_template = accelerator_test_float32(make_point_samples_template())

        cpu_standard = PointSamples_interpolation(point_template, deepcopy(standard_input32), catalog, itpScatter)
        cpu_smoothing = PointSamples_interpolation(point_template, deepcopy(smoothing_input32), catalog, itpScatter)

        gpu_standard = PointSamples_interpolation(
            to_device(point_template),
            to_device(standard_input32),
            catalog,
            itpScatter,
        )
        gpu_smoothing = PointSamples_interpolation(
            to_device(point_template),
            to_device(smoothing_input32),
            catalog,
            itpScatter,
        )
        @test same_coordinates(gpu_standard.grids...)
        @test same_coordinates(gpu_smoothing.grids...)
        synchronize()

        host_standard = accelerator_test_host_bundle(gpu_standard, to_host)
        host_smoothing = accelerator_test_host_bundle(gpu_smoothing, to_host)
        accelerator_test_bundle_equal(host_standard, cpu_standard; atol, rtol)
        accelerator_test_bundle_equal(host_smoothing, cpu_smoothing; atol, rtol)
        accelerator_test_bundle_equal(host_smoothing, host_standard; atol, rtol)
    end

    @testset "$name -- PointSamples interpolation -- analytic linear field" begin
        analytic_input, catalog, _ = make_uniform_cloud_3d(12; eta = 1.2, variable_h = true)
        input32 = accelerator_test_float32(analytic_input)
        point_template = accelerator_test_float32(make_analytic_point_samples())

        for strategy in (itpGather, itpScatter)
            cpu_result = PointSamples_interpolation(point_template, deepcopy(input32), catalog, strategy)
            gpu_result = PointSamples_interpolation(
                to_device(point_template),
                to_device(input32),
                catalog,
                strategy,
            )
            @test same_coordinates(gpu_result.grids...)
            synchronize()

            host_result = accelerator_test_host_bundle(gpu_result, to_host)
            accelerator_test_bundle_equal(host_result, cpu_result; atol, rtol)
        end
    end

    # ── 4. LineSamples interpolation ───────────────────────────────────── #

    @testset "$name -- LineSamples interpolation -- CPU consistency" begin
        source_input, catalog = accelerator_test_line_fixture()
        line_template = accelerator_test_float32(make_line_samples_template())
        cpu_result = LineSamples_interpolation(line_template, deepcopy(source_input), catalog)

        gpu_result = LineSamples_interpolation(
            to_device(line_template),
            to_device(source_input),
            catalog,
        )
        @test same_coordinates(gpu_result.grids...)
        synchronize()

        host_result = accelerator_test_host_bundle(gpu_result, to_host)
        accelerator_test_bundle_equal(host_result, cpu_result; atol, rtol)
    end

    @testset "$name -- LineSamples interpolation -- externally supplied LBVH" begin
        source_input, catalog = accelerator_test_line_fixture()
        line_template = accelerator_test_float32(make_line_samples_template())

        auto_input = to_device(source_input)
        auto_result = LineSamples_interpolation(to_device(line_template), auto_input, catalog)

        manual_input = to_device(source_input)
        manual_lbvh = LinearBVH!(manual_input)
        manual_result = LineSamples_interpolation(
            to_device(line_template),
            manual_input,
            catalog,
            manual_lbvh,
        )
        @test same_coordinates(auto_result.grids...)
        @test same_coordinates(manual_result.grids...)
        synchronize()

        host_auto = accelerator_test_host_bundle(auto_result, to_host)
        host_manual = accelerator_test_host_bundle(manual_result, to_host)
        accelerator_test_bundle_equal(host_manual, host_auto; atol, rtol)
    end

    @testset "$name -- LineSamples interpolation -- smoothing-volume consistency" begin
        standard_input, smoothing_input, catalog = make_smoothing_volume_line_interpolation_fixture()
        standard_input32 = accelerator_test_float32(standard_input)
        smoothing_input32 = accelerator_test_float32(smoothing_input)
        line_template = accelerator_test_float32(make_line_samples_template())

        cpu_standard = LineSamples_interpolation(line_template, deepcopy(standard_input32), catalog)
        cpu_smoothing = LineSamples_interpolation(line_template, deepcopy(smoothing_input32), catalog)

        gpu_standard = LineSamples_interpolation(
            to_device(line_template),
            to_device(standard_input32),
            catalog,
        )
        gpu_smoothing = LineSamples_interpolation(
            to_device(line_template),
            to_device(smoothing_input32),
            catalog,
        )
        @test same_coordinates(gpu_standard.grids...)
        @test same_coordinates(gpu_smoothing.grids...)
        synchronize()

        host_standard = accelerator_test_host_bundle(gpu_standard, to_host)
        host_smoothing = accelerator_test_host_bundle(gpu_smoothing, to_host)
        accelerator_test_bundle_equal(host_standard, cpu_standard; atol, rtol)
        accelerator_test_bundle_equal(host_smoothing, cpu_smoothing; atol, rtol)
        accelerator_test_bundle_equal(host_smoothing, host_standard; atol, rtol)
    end

    # ── 5. StructuredGrid interpolation ────────────────────────────────── #

    @testset "$name -- StructuredGrid interpolation -- CPU consistency" begin
        for (coord, template) in (
            (Cartesian, make_structured_grid_template()),
            (Cylindrical, make_cylindrical_grid_template()),
            (Spherical, make_spherical_grid_template()),
        )
            structured_template = accelerator_test_float32(template)

            for strategy in (itpGather, itpScatter)
                source_input, catalog = accelerator_test_grid_fixture()
                cpu_result = StructuredGrid_interpolation(
                    coord,
                    structured_template,
                    deepcopy(source_input),
                    catalog,
                    strategy,
                )
                gpu_result = StructuredGrid_interpolation(
                    coord,
                    to_device(structured_template),
                    to_device(source_input),
                    catalog,
                    strategy,
                )
                @test same_coordinates(gpu_result.grids...)
                synchronize()

                host_result = accelerator_test_host_bundle(gpu_result, to_host)
                accelerator_test_bundle_equal(host_result, cpu_result; atol, rtol)
            end
        end
    end

    @testset "$name -- StructuredGrid interpolation -- analytic linear field" begin
        analytic_input, catalog, _ = make_uniform_cloud_3d(12; eta = 1.2, variable_h = true)
        input32 = accelerator_test_float32(analytic_input)

        for (coord, template) in (
            (Cartesian, make_analytic_structured_grid()),
            (Cylindrical, make_analytic_cylindrical_grid()),
            (Spherical, make_analytic_spherical_grid()),
        )
            structured_template = accelerator_test_float32(template)

            for strategy in (itpGather, itpScatter)
                cpu_result = StructuredGrid_interpolation(
                    coord,
                    structured_template,
                    deepcopy(input32),
                    catalog,
                    strategy,
                )
                gpu_result = StructuredGrid_interpolation(
                    coord,
                    to_device(structured_template),
                    to_device(input32),
                    catalog,
                    strategy,
                )
                @test same_coordinates(gpu_result.grids...)
                synchronize()

                host_result = accelerator_test_host_bundle(gpu_result, to_host)
                accelerator_test_bundle_equal(host_result, cpu_result; atol, rtol)
            end
        end
    end

    return nothing
end

######################################################################################

#  Test: SPH Interpolation -- Constructors, Traversal, and Physical Checks
#  What this file tests
#  End-to-end validation of the core KernelInterpolation pipeline:
#  1. Core `build_input` constructor
#     • Builds `InterpolationInput` and `InterpolationCatalog` directly from
#        already-materialized particle columns.
#     • Verifies particle count, element-type promotion, catalog slot lookup,
#        quantity storage, and error paths for missing requested columns.
#  2. BVH traversal interpolation vs brute-force references
#     • Density, number density, quantity, gradient, divergence, and curl
#        interpolation (3D) against O(N) brute-force baselines for all three
#        strategies (Gather, Scatter).
#     • Line-integrated column-density and quantity interpolation.
#  3. Physical sanity checks
#     • Divergence and curl of a uniform vector field must vanish to machine
#        epsilon.
#  Brute-force reference implementations live in `interpolation_test_common.jl`,
#  which is included by this file.

######################################################################################
using Test
using Random
using Partia

# ========================== Internal API imports ============================ #

using Partia.KernelInterpolation:
    _general_quantity_interpolate_kernel,
    _line_integrated_quantities_interpolate_kernel

# ========================== Module aliases ================================== #

ki_mod = Partia.KernelInterpolation

# ========================== Shared includes ================================= #

@static if !isdefined(@__MODULE__, :support_radius)
    include("interpolation_test_common.jl")
end

general_scalar_catalog() = to_concise_catalog(InterpolationCatalog(
    Val(3);
    scalar_names = (:q1,),
    scalar_slots = (1,),
))

general_vector_catalog() = to_concise_catalog(InterpolationCatalog(
    Val(3);
    scalar_names = (:q1,),
    scalar_slots = (1,),
    grad_names = (:q2,),
    grad_slots = (2,),
    div_names = (:q,),
    div_slots = ((1, 2, 3),),
    curl_names = (:q,),
    curl_slots = ((1, 2, 3),),
))

@inline general_interpolate(input, point, ha, LBVH, catalog, strategy) =
    strategy === itpScatter ?
        _general_quantity_interpolate_kernel(input, point, LBVH, catalog) :
        _general_quantity_interpolate_kernel(input, point, ha, LBVH, catalog)

# ============================== Test body =================================== #

# ── 0. Empty and no-neighbor behavior ───────────────────────────────────── #

@testset "Traversal interpolation -- empty and no-neighbor behavior" begin
    empty_input, empty_bvh = make_empty_input()
    ref3d = (0.5, 0.5, 0.5)
    ha = 0.05
    scalar_catalog = general_scalar_catalog()

    for strategy in (itpGather, itpScatter)
        scalars, gradients, divergences, curls = general_interpolate(empty_input, ref3d, ha, empty_bvh, scalar_catalog, strategy)

        @test all(isnan, scalars)
        @test isempty(gradients)
        @test isempty(divergences)
        @test isempty(curls)
    end

    rng = Xoshiro(0xDEAD)
    input, LBVH = random_input_3d(rng, 20)
    far_point = (2.0, 2.0, 2.0)
    tiny_ha = 1.0e-4
    vector_catalog = general_vector_catalog()

    for strategy in (itpGather, itpScatter)
        scalars, gradients, divergences, curls = general_interpolate(input, far_point, tiny_ha, LBVH, vector_catalog, strategy)

        @test all(isnan, scalars)
        @test all(g -> all(isnan, g), gradients)
        @test all(isnan, divergences)
        @test all(c -> all(isnan, c), curls)
    end
end


# ── 1a. InterpolationInput -- core build_input constructor ─────────────── #

@testset "InterpolationInput -- core build_input constructor" begin
    x = Float32[0.0, 1.0, 2.0]
    y = Float32[1.0, 2.0, 3.0]
    z = Float32[2.0, 3.0, 4.0]
    h = Float32[0.2, 0.25, 0.3]
    rho = Float64[1.0, 1.1, 0.9]
    m = Float32[0.5, 0.6, 0.7]
    P = Float32[10.0, 11.0, 12.0]
    vx = Float32[0.1, 0.0, -0.1]
    vy = Float32[0.0, 0.1, 0.0]
    vz = Float32[-0.1, 0.0, 0.1]
    Bx = Float32[1.0, 1.1, 1.2]
    By = Float32[1.2, 1.3, 1.4]
    Bz = Float32[1.4, 1.5, 1.6]

    input, catalog = build_input(
        x, y, z, m, h, rho, (P, vx, vy, vz, Bx, By, Bz);
        column_names = (:P, :vx, :vy, :vz, :Bx, :By, :Bz),
        scalars = (:P,),
        gradients = (:P,),
        divergences = (:v,),
        curls = (:B,),
    )

    @test input.Npart == 3
    @test eltype(get_xcoord(input)) === Float64
    @test length(input.quant) == 7

    @test ki_mod.scalar_index(catalog, :P) == 1
    @test ki_mod.grad_slot(catalog, :P) == 1
    @test ki_mod.div_slots(catalog, :v) == (2, 3, 4)
    @test ki_mod.curl_slots(catalog, :B) == (5, 6, 7)

    @test ki_mod.ordered_quantity_names(catalog)[1] == :P
    @test length(ki_mod.ordered_quantity_names(catalog)) == 8

    @test all(input.quant[1] .== Float64.(P))
    @test all(input.quant[2] .== Float64.(vx))
    @test all(input.quant[7] .== Float64.(Bz))

    @test_throws KeyError build_input(
        x, y, z, m, h, rho, (P, vx, vy, vz, Bx, By);
        column_names = (:P, :vx, :vy, :vz, :Bx, :By),
        scalars = (),
        gradients = (),
        divergences = (),
        curls = (:B,),
    )

    input_2d, catalog_2d = build_input(
        x, y, m, h, rho, (P, vx, vy);
        column_names = (:P, :vx, :vy),
        scalars = (:P,),
        gradients = (:P,),
        divergences = (:v,),
    )

    @test input_2d isa InterpolationInput{2, Float64}
    @test catalog_2d isa InterpolationCatalog{2, 1, 1, 1, 0, 4}
    @test ki_mod.div_slots(catalog_2d, :v) == (2, 3)
    @test ki_mod.ordered_quantity_names(catalog_2d) == (:P, :∇Pˣ, :∇Pʸ, Symbol("∇⋅v"))

    smoothing_input_2d, smoothing_catalog_2d = build_input(
        1.2f0, x, y, m, h, (P, vx, vy);
        column_names = (:P, :vx, :vy),
        scalars = (:P,),
        gradients = (:P,),
        divergences = (:v,),
    )

    @test smoothing_input_2d isa InterpolationSmoothingVolumeInput{2, Float32}
    @test smoothing_catalog_2d isa InterpolationCatalog{2, 1, 1, 1, 0, 4}
end


# ── 1b. InterpolationInput -- direct array constructor ─────────────────── #

@testset "InterpolationInput -- direct array constructor" begin
    x = Float32[0.0, 1.0, 2.0]
    y = Float32[1.0, 2.0, 3.0]
    z = Float32[2.0, 3.0, 4.0]
    h = Float32[0.2, 0.25, 0.3]
    rho = Float64[1.0, 1.1, 0.9]
    m = fill(0.42f0, 3)
    P = Float32[10.0, 11.0, 12.0]
    vx = Float32[0.1, 0.0, -0.1]
    vy = Float32[0.0, 0.1, 0.0]
    vz = Float32[-0.1, 0.0, 0.1]
    Bx = Float32[1.0, 1.1, 1.2]
    By = Float32[1.2, 1.3, 1.4]
    Bz = Float32[1.4, 1.5, 1.6]

    input, catalog = build_input(
        x, y, z, m, h, rho, (P, vx, vy, vz, Bx, By, Bz);
        column_names = (:P, :vx, :vy, :vz, :Bx, :By, :Bz),
        scalars = (:P,),
        gradients = (:P,),
        divergences = (:v,),
        curls = (:B,),
    )

    @test input.Npart == 3
    @test eltype(get_xcoord(input)) === Float64
    @test length(input.quant) == 7
    @test all(input.m .== fill(Float64(0.42f0), 3))

    @test ki_mod.scalar_index(catalog, :P) == 1
    @test ki_mod.grad_slot(catalog, :P) == 1
    @test ki_mod.div_slots(catalog, :v) == (2, 3, 4)
    @test ki_mod.curl_slots(catalog, :B) == (5, 6, 7)

    @test_throws KeyError build_input(
        x, y, z, m, h, rho, (vx, vy, vz, Bx, By, Bz);
        column_names = (:vx, :vy, :vz, :Bx, :By, :Bz),
        scalars = (:P,),
        gradients = (),
        divergences = (),
        curls = (),
    )
end


# ── 2a. Traversal -- scalar quantity ───────────────────────────────────── #

@testset "Traversal interpolation -- scalar quantity (3D)" begin
    rng = MersenneTwister(0xBADA55)
    input, LBVH = random_input_3d(rng, 200)
    reference_point = (0.4, 0.35, 0.25)
    ha = 0.12
    catalog = general_scalar_catalog()

    for strategy in (itpGather, itpScatter)
        scalars, _, _, _ = general_interpolate(input, reference_point, ha, LBVH, catalog, strategy)
        qty = scalars[1]

        @test isapprox(qty, brute_quantity(input, reference_point, ha, 1, strategy); atol = 1e-10, rtol = 1e-8)
    end
end


# ── 2b. Traversal -- gradients and vector operators (3D) ───────────────── #

@testset "Traversal interpolation -- gradients and vector operators (3D)" begin
    rng = MersenneTwister(0xC0FFEE)
    input, LBVH = random_input_3d(rng, 80)
    reference_points = ((0.2, 0.3, 0.4), (0.7, 0.2, 0.1))
    ha_values = (0.05, 0.12)
    catalog = general_vector_catalog()

    for reference_point in reference_points, ha in ha_values, strategy in (itpGather, itpScatter)
        _, gradients, divergences, curls = general_interpolate(input, reference_point, ha, LBVH, catalog, strategy)
        grad_A = gradients[1]
        div_A = divergences[1]
        curl_A = curls[1]

        @test approx_with_nan(grad_A, brute_gradient_quantity(input, reference_point, ha, 2, strategy); atol = 5e-10, rtol = 1e-8)
        @test approx_with_nan(div_A, brute_divergence(input, reference_point, ha, (1, 2, 3), strategy); atol = 5e-10, rtol = 1e-8)
        @test approx_with_nan(curl_A, brute_curl(input, reference_point, ha, (1, 2, 3), strategy); atol = 5e-10, rtol = 1e-8)
    end
end


# ── 2c. Traversal -- line-integrated quantity ──────────────────────────── #

@testset "Traversal interpolation -- line-integrated quantity" begin
    rng = MersenneTwister(0xF00D)
    input, LBVH = random_input_line_integrated(rng, 150)
    origin = (0.2, 0.8, 0.0)
    direction = (0.0, 0.0, 1.0)
    ha = 0.08

    qty = _line_integrated_quantities_interpolate_kernel(input, origin, direction, LBVH, (1,), (true,))[1]

    @test approx_with_nan(qty, brute_line_integrated_quantity(input, origin, direction, ha, 1, itpScatter); atol = 1e-10, rtol = 1e-8)
end


# ── 3. Divergence & curl vanish for uniform field ───────────────────────── #

@testset "Uniform field -- divergence = 0, curl = 0" begin
    n = 4
    x = [0.0, 0.05, 0.11, -0.08]
    y = [0.02, -0.03, 0.04, 0.01]
    z = [0.0, 0.01, -0.02, 0.03]
    m = fill(1.0, n)
    h = fill(0.12, n)
    rho = fill(1.0, n)
    vx = fill(1.0, n)
    vy = fill(-2.0, n)
    vz = fill(0.5, n)

    input = InterpolationInput((x, y, z), m, h, rho, (vx, vy, vz); smoothed_kernel = typeof(kern))
    LBVH_local = LinearBVH!(input)
    catalog = to_concise_catalog(InterpolationCatalog(
        Val(3);
        div_names = (:v,),
        div_slots = ((1, 2, 3),),
        curl_names = (:v,),
        curl_slots = ((1, 2, 3),),
    ))

    reference_point = (x[1], y[1], z[1])
    ha = h[1]

    for strategy in (itpGather, itpScatter)
        _, _, divergences, curls = general_interpolate(input, reference_point, ha, LBVH_local, catalog, strategy)
        divv = divergences[1]
        curlv = curls[1]

        @test isapprox(divv, 0.0; atol = 1e-12, rtol = 1e-10)
        @test isapprox(curlv[1], 0.0; atol = 1e-12, rtol = 1e-10)
        @test isapprox(curlv[2], 0.0; atol = 1e-12, rtol = 1e-10)
        @test isapprox(curlv[3], 0.0; atol = 1e-12, rtol = 1e-10)
    end
end

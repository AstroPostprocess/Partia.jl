######################################################################################

#  Test: Type Stability — Core Numerical Kernels and Hot Helpers
#  What this file tests
#  Representative `@inferred` checks for concrete, performance-critical code
#  paths that should stay type-stable:
#  1. SPH kernel evaluation
#     • `Smoothed_kernel_function*`
#     • `Smoothed_gradient_kernel_function`
#     • `line_integrated_kernel_function*`
#  2. Neighbor-search query helpers
#     • `LBVH_probe_neighbors`
#     • `LBVH_find_nearest`
#     • `LBVH_find_nearest_h`
#     • `LBVH_query!`
#  3. Single-point interpolation kernels
#     • fused scalar / gradient / divergence / curl interpolation
#  4. Line-integrated interpolation kernels
#     • single- and multi-quantity interpolation
#  Deliberately excluded here:
#  • I/O, logging, and orchestration-heavy wrappers
#  • APIs that intentionally return flexible container shapes
#  • broad generic method sweeps; only representative concrete inputs are used

######################################################################################
using Test
using StaticArrays
using Partia
using Partia.KernelInterpolation:
    _general_quantity_interpolate_kernel,
    _line_integrated_quantities_interpolate_kernel


# ========================== Fixture builders ================================ #

function make_type_stability_input_3d()
    kern = M4_spline()
    x = Float64[0.10, 0.22, 0.34, 0.46, 0.58, 0.70]
    y = Float64[0.15, 0.28, 0.20, 0.52, 0.44, 0.68]
    z = Float64[0.12, 0.18, 0.36, 0.48, 0.62, 0.40]
    m = Float64[0.4, 0.45, 0.42, 0.38, 0.41, 0.43]
    h = Float64[0.16, 0.18, 0.17, 0.19, 0.16, 0.18]
    ρ = Float64[1.0, 1.1, 0.95, 1.05, 1.02, 0.98]
    q1 = Float64[10.0, 11.5, 9.0, 12.0, 13.5, 8.5]
    q2 = Float64[0.1, -0.2, 0.3, -0.1, 0.2, -0.3]
    q3 = Float64[0.0, 0.25, -0.15, 0.2, -0.05, 0.1]
    input = InterpolationInput((x, y, z), m, h, ρ, (q1, q2, q3); smoothed_kernel = typeof(kern))
    lbvh = LinearBVH!(input)
    return input, lbvh
end

function make_type_stability_smoothing_input_3d()
    kern = M4_spline()
    hfact = 1.2
    x = Float64[0.10, 0.22, 0.34, 0.46, 0.58, 0.70]
    y = Float64[0.15, 0.28, 0.20, 0.52, 0.44, 0.68]
    z = Float64[0.12, 0.18, 0.36, 0.48, 0.62, 0.40]
    m = Float64[0.4, 0.45, 0.42, 0.38, 0.41, 0.43]
    h = Float64[0.16, 0.18, 0.17, 0.19, 0.16, 0.18]
    q1 = Float64[10.0, 11.5, 9.0, 12.0, 13.5, 8.5]
    q2 = Float64[0.1, -0.2, 0.3, -0.1, 0.2, -0.3]
    q3 = Float64[0.0, 0.25, -0.15, 0.2, -0.05, 0.1]
    input = InterpolationSmoothingVolumeInput(hfact, (x, y, z), m, h, (q1, q2, q3); smoothed_kernel = typeof(kern))
    lbvh = LinearBVH!(input)
    return input, lbvh
end

function make_type_stability_input_line_integrated()
    kern = M4_spline()
    x = Float64[0.12, 0.26, 0.38, 0.51, 0.64]
    y = Float64[0.08, 0.30, 0.22, 0.57, 0.41]
    z = Float64[0.10, 0.24, 0.48, 0.36, 0.62]
    m = Float64[0.5, 0.43, 0.47, 0.39, 0.44]
    h = Float64[0.15, 0.17, 0.16, 0.18, 0.15]
    ρ = Float64[1.0, 1.08, 0.97, 1.02, 1.05]
    q1 = Float64[2.0, 2.5, 3.0, 3.5, 4.0]
    input = InterpolationInput((x, y, z), m, h, ρ, (q1,); smoothed_kernel = typeof(kern))
    lbvh = LinearBVH!(input)
    return input, lbvh
end

function make_type_stability_smoothing_input_line_integrated()
    kern = M4_spline()
    hfact = 1.2
    x = Float64[0.12, 0.26, 0.38, 0.51, 0.64]
    y = Float64[0.08, 0.30, 0.22, 0.57, 0.41]
    z = Float64[0.10, 0.24, 0.48, 0.36, 0.62]
    m = Float64[0.5, 0.43, 0.47, 0.39, 0.44]
    h = Float64[0.15, 0.17, 0.16, 0.18, 0.15]
    q1 = Float64[2.0, 2.5, 3.0, 3.5, 4.0]
    input = InterpolationSmoothingVolumeInput(hfact, (x, y, z), m, h, (q1,); smoothed_kernel = typeof(kern))
    lbvh = LinearBVH!(input)
    return input, lbvh
end

function make_type_stability_catalog()
    return to_concise_catalog(InterpolationCatalog(
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
end


# ============================== Test body =================================== #

# ── 1. Kernel evaluation ────────────────────────────────────────────── #

@testset "Type stability — kernel evaluation" begin
    kern = M4_spline()
    @inferred Smoothed_kernel_function(typeof(kern), 0.12, 0.20, Val(3))
    @inferred Smoothed_kernel_function(typeof(kern), (0.1, 0.2, 0.3), (0.2, 0.0, 0.4), 0.20)
    @inferred Smoothed_gradient_kernel_function(typeof(kern), 0.05, -0.03, 0.08, 0.20)
    @inferred Smoothed_gradient_kernel_function(typeof(kern), (0.1, 0.2, 0.3), (0.2, 0.0, 0.4), 0.20)
    @inferred line_integrated_kernel_function(typeof(kern), 0.07, 0.20)
    @inferred line_integrated_kernel_function(typeof(kern), (0.1, 0.2), (0.3, 0.4), 0.20)
end

# ── 2. Neighbor-search query helpers ────────────────────────────────── #

@testset "Type stability — LBVH query helpers" begin
    input, lbvh = make_type_stability_input_3d()
    point = (0.33, 0.27, 0.31)
    radius = 0.22
    pool = zeros(Int, input.Npart)

    @inferred LBVH_probe_neighbors(lbvh, point, radius)
    @inferred LBVH_find_nearest(lbvh, point)
    @inferred LBVH_find_nearest_h(lbvh, point)
    @inferred LBVH_query!(pool, lbvh, point, radius)
end

# ── 3. Single-point interpolation kernels ───────────────────────────── #

@testset "Type stability — single-point interpolation kernels" begin
    input, lbvh = make_type_stability_input_3d()
    point = (0.33, 0.27, 0.31)
    ha = 0.19
    catalog = make_type_stability_catalog()

    @inferred _general_quantity_interpolate_kernel(input, point, ha, lbvh, catalog)
    @inferred _general_quantity_interpolate_kernel(input, point, lbvh, catalog)

    smoothing_input, smoothing_lbvh = make_type_stability_smoothing_input_3d()
    @inferred _general_quantity_interpolate_kernel(smoothing_input, point, ha, smoothing_lbvh, catalog)
    @inferred _general_quantity_interpolate_kernel(smoothing_input, point, smoothing_lbvh, catalog)
end

# ── 4. Line-integrated interpolation kernels ────────────────────────── #

@testset "Type stability — line-integrated interpolation kernels" begin
    input, lbvh = make_type_stability_input_line_integrated()
    origin = (0.30, 0.35, 0.00)
    direction = (0.0, 0.0, 1.0)

    @inferred _line_integrated_quantities_interpolate_kernel(input, origin, direction, lbvh, (1,), (true,))

    smoothing_input, smoothing_lbvh = make_type_stability_smoothing_input_line_integrated()
    @inferred _line_integrated_quantities_interpolate_kernel(smoothing_input, origin, direction, smoothing_lbvh, (1,), (true,))
end

######################################################################################

#  Test: Numerical Tools
#  What this file tests
#  Validates general-purpose numerical utilities:
#  1. Array construction helpers
#     • 2D and 3D meshgrid construction.
#  2. NaN-safe statistics
#     • Mean, extrema, and standard deviation with NaN entries ignored.
#  3. Geometry and array-order helpers
#     • Euclidean distance utilities.
#     • Inverse permutation and complex-array maximum absolute value.

######################################################################################
using Test
using Partia

# ============================== Test body =================================== #

# ── 1. meshgrid construction ─────────────────────────────────────────── #

@testset "meshgrid -- 2D and 3D" begin
    xs = [1.0, 2.0, 3.0]
    ys = [10.0, 20.0]

    X, Y = meshgrid(xs, ys)
    @test size(X) == (3, 2)
    @test size(Y) == (3, 2)
    @test X[:, 1] == xs
    @test X[:, 2] == xs
    @test Y[1, :] == ys
    @test Y[3, :] == ys

    zs = [100.0]
    X3, Y3, Z3 = meshgrid(xs, ys, zs)
    @test size(X3) == (3, 2, 1)
    @test size(Y3) == (3, 2, 1)
    @test size(Z3) == (3, 2, 1)
end

# ── 2. NaN-safe statistics ───────────────────────────────────────────── #

@testset "NaN-safe statistics" begin
    A = [1.0, NaN, 3.0, NaN, 5.0]

    @test nanmean(A) ≈ 3.0
    @test nanmaximum(A) ≈ 5.0
    @test nanminimum(A) ≈ 1.0
    @test isnan(nanmean([NaN, NaN]))

    B = [2.0, 4.0, 6.0]
    @test nanmean(B) ≈ 4.0
    @test nanstd(B) ≈ 2.0
end

# ── 3. Euclidean distance ────────────────────────────────────────────── #

@testset "Euclidean distance" begin
    x = [0.0, 3.0]
    y = [0.0, 4.0]
    z = [0.0, 0.0]

    d3 = Euclidean_distance(x, y, z, (0.0, 0.0, 0.0))
    @test d3[1] ≈ 0.0
    @test d3[2] ≈ 5.0

    d2 = Euclidean_distance(x, y, (0.0, 0.0))
    @test d2[1] ≈ 0.0
    @test d2[2] ≈ 5.0
end

# ── 4. Permutation helpers ───────────────────────────────────────────── #

@testset "invert_order -- inverse permutation identity" begin
    order = [3, 1, 4, 2]
    inv_order = invert_order(order)
    v = [10, 20, 30, 40]
    @test v[order][inv_order] == v
end

# ── 5. Array norms ───────────────────────────────────────────────────── #

@testset "maxabs -- complex array" begin
    A = ComplexF64[1 + 2im, 3 + 4im, -5 + 0im]
    @test maxabs(A) ≈ 5.0
end

######################################################################################

# Unified LinearBVH construction and traversal regression tests.

######################################################################################
using Test
using Random
using Partia
using Partia.LinearBoundingVolumeHierarchy

const lbvh_mod = Partia.LinearBoundingVolumeHierarchy

function encoding(::Val{2}, n, seed)
    rng = MersenneTwister(seed)
    MortonEncoding(rand(rng, n), rand(rng, n))
end

function encoding(::Val{3}, n, seed)
    rng = MersenneTwister(seed)
    MortonEncoding(rand(rng, n), rand(rng, n), rand(rng, n))
end

@inline right_child(lbvh, node) = lbvh.escape[Int(lbvh.left[Int(node)])]

function visit_nodes(lbvh)
    visited = Int[]
    node = Int32(1)
    while !iszero(node)
        push!(visited, Int(node))
        node = lbvh_mod.is_leaf_id(node, lbvh.nleaf) ? lbvh.escape[Int(node)] : lbvh.left[Int(node)]
    end
    visited
end

function brute_force(enc, point, radius)
    radius2 = radius^2
    sort!([i for i in eachindex(enc.coord[1]) if
        sum((enc.coord[d][i] - point[d])^2 for d in eachindex(point)) <= radius2])
end

@testset "LinearBVH -- unified topology" begin
    for D in (Val(2), Val(3)), n in 1:30
        enc = encoding(D, n, 1000 + n)
        h = n == 1 ? [0.1] : collect(range(0.01, 0.2; length=n))
        lbvh = LinearBVH(enc, h)
        total = 2n - 1

        @test lbvh.nleaf == n
        @test length(lbvh.left) == n - 1
        @test length(lbvh.escape) == total
        @test length(lbvh.scale) == total
        @test all(length(lbvh.aabb.min[d]) == total for d in 1:length(enc.coord))
        @test sort(visit_nodes(lbvh)) == collect(1:total)
        @test lbvh.escape[1] == 0

        leaf_nodes = n:total
        @test lbvh.scale[leaf_nodes] == h
        for d in 1:length(enc.coord)
            @test lbvh.aabb.min[d][leaf_nodes] == enc.coord[d]
            @test lbvh.aabb.max[d][leaf_nodes] == enc.coord[d]
        end

        for node in 1:n-1
            left = Int(lbvh.left[node])
            right = Int(right_child(lbvh, Int32(node)))
            @test 1 <= left <= total
            @test 1 <= right <= total
            @test lbvh.scale[node] == max(lbvh.scale[left], lbvh.scale[right])
            for d in 1:length(enc.coord)
                @test lbvh.aabb.min[d][node] == min(lbvh.aabb.min[d][left], lbvh.aabb.min[d][right])
                @test lbvh.aabb.max[d][node] == max(lbvh.aabb.max[d][left], lbvh.aabb.max[d][right])
            end
        end
    end
end

@testset "LinearBVH -- identical Morton codes" begin
    for D in (Val(2), Val(3)), n in 1:24
        dim = D isa Val{2} ? 2 : 3
        coords = ntuple(_ -> fill(0.5, n), dim)
        enc = MortonEncoding(coords...)
        lbvh = LinearBVH(enc, ones(n))
        @test sort(visit_nodes(lbvh)) == collect(1:2n-1)
        @test all(==(0.5), (lbvh.aabb.min[d][1] for d in 1:dim))
    end
end

@testset "LinearBVH -- point queries match brute force" begin
    rng = MersenneTwister(42)
    for D in (Val(2), Val(3)), n in (1, 2, 7, 32)
        dim = D isa Val{2} ? 2 : 3
        enc = encoding(D, n, 2000 + n)
        lbvh = LinearBVH(enc, fill(0.1, n))
        pool = Vector{Int}(undef, n)
        for _ in 1:20
            point = ntuple(_ -> rand(rng), dim)
            radius = rand(rng) * 0.5
            result = LBVH_query!(pool, lbvh, point, radius)
            @test sort(collect(valid_indices(result))) == brute_force(enc, point, radius)
        end
    end
end

@testset "LinearBVH -- point and line traversal" begin
    enc = encoding(Val(3), 64, 99)
    h = rand(MersenneTwister(8), 64) .* 0.15 .+ 0.01
    lbvh = LinearBVH(enc, h)
    point = (0.4, 0.5, 0.6)
    K = 2.0
    expected = sort([i for i in 1:64 if sum((enc.coord[d][i] - point[d])^2 for d in 1:3) <= (K*h[i])^2])
    actual = Int[]
    leaf = 0; d2 = 0.0; hb = 0.0
    @LBVH_scatter_point_traversal lbvh point K leaf d2 hb push!(actual, leaf)
    @test sort(actual) == expected

    origin = (0.5, 0.5, 0.5)
    direction = (1.0, 0.0, 0.0)
    radius2 = 0.1^2
    expected_line = sort([i for i in 1:64 if
        lbvh_mod._squared_distance_point_line(ntuple(d -> enc.coord[d][i], 3), origin, direction) <= radius2])
    empty!(actual)
    @LBVH_gather_line_traversal lbvh origin direction radius2 leaf d2 push!(actual, leaf)
    @test sort(actual) == expected_line
end

@testset "LinearBVH -- finite leaf AABB traversal" begin
    enc = MortonEncoding([0.5], [1.5])
    leaf_min = ([0.0], [1.0])
    leaf_max = ([1.0], [2.0])
    lbvh = LinearBVH(enc, [0.1], leaf_min, leaf_max)

    # The query is inside the box but far from the encoded leaf point.
    point = (0.05, 1.05)
    point_hits = Int[]
    leaf = 0
    d2 = 0.0
    @LBVH_gather_point_traversal lbvh point 0.01^2 leaf d2 push!(point_hits, leaf)
    @test point_hits == [1]
    @test d2 == 0.0

    # This line crosses the box while remaining far from its center point.
    origin = (0.0, 1.05)
    direction = (1.0, 0.0)
    line_hits = Int[]
    @LBVH_gather_line_traversal lbvh origin direction 0.01^2 leaf d2 push!(line_hits, leaf)
    @test line_hits == [1]
    @test d2 == 0.0

    scatter_hits = Int[]
    hb = 0.0
    @LBVH_scatter_line_traversal lbvh origin direction 1.0 leaf d2 hb push!(scatter_hits, leaf)
    @test scatter_hits == [1]
    @test hb == 0.1
end

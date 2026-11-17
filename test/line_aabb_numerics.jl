# Run independently: julia --project=. test/line_aabb_numerics.jl
using Test, Random, Partia

module LineAABBNumericsTests
using Test, Random, Partia
const LB = Partia.LinearBoundingVolumeHierarchy
storage(x) = map(v -> [v], x)
distance(o,v,a,b) = LB._squared_distance_line_aabb(o,v,storage(a),storage(b),1)
prune(o,v,a,b,r2,d2) = LB._line_aabb_can_prune(o,v,storage(a),storage(b),1,r2,d2)

# Independent oracle: enumerate every quadratic interval of the ORIGINAL box.
# High precision is needed for Float64 subnormal directions and extreme scales.
function reference(o,v,a,b)
    p,d,lo,hi = BigFloat.(o),BigFloat.(v),BigFloat.(a),BigFloat.(b)
    D = length(o)
    cuts = sort!([(x-p[k])/d[k] for k in 1:D if d[k]!=0 for x in (lo[k],hi[k])])
    isempty(cuts) && return sum(max(lo[k]-p[k],p[k]-hi[k],0)^2 for k in 1:D)
    edges = vcat(-BigFloat(Inf),cuts,BigFloat(Inf))
    best = BigFloat(Inf)
    for j in 1:length(edges)-1
        l,h = edges[j],edges[j+1]
        t = isinf(l) ? h-max(1,abs(h)) : isinf(h) ? l+max(1,abs(l)) : (l+h)/2
        aa = bb = BigFloat(0)
        for k in 1:D
            x = p[k]+t*d[k]
            if x<lo[k] || x>hi[k]
                bound = x<lo[k] ? lo[k] : hi[k]
                aa += d[k]^2
                bb += d[k]*(p[k]-bound)
            end
        end
        u = aa==0 ? t : clamp(-bb/aa,l,h)
        best = min(best,sum(max(lo[k]-p[k]-u*d[k],p[k]+u*d[k]-hi[k],0)^2 for k in 1:D))
    end
    return best
end

function check_enclosure(o,a,b)
    c,e = LB._line_aabb_local_box(o,storage(a),storage(b),1)
    for k in eachindex(o)
        @test BigFloat(c[k])-BigFloat(e[k]) <= BigFloat(a[k])-BigFloat(o[k])
        @test BigFloat(c[k])+BigFloat(e[k]) >= BigFloat(b[k])-BigFloat(o[k])
    end
end

@testset "Line-AABB numerical safety" begin
    setprecision(BigFloat,4096) do
        @testset "Local enclosure and original rounding regression" begin
            for T in (Float32,Float64), D in (2,3,4)
                x = T(1e8)
                a = ntuple(k -> k==1 ? x : zero(T),D)
                b = ntuple(k -> k==1 ? nextfloat(x) : zero(T),D)
                o = b
                v = ntuple(k -> T(k==2),D)
                check_enclosure(o,a,b)
                @test distance(o,v,a,b) == 0
                @test distance(o,v,b,b) == 0
                @test !prune(o,v,a,b,one(T),T(16)) # even with the OLD erroneous value
                for x in (nextfloat(zero(T)),floatmin(T),one(T),floatmax(T)/T(2))
                    aa = ntuple(_ -> -x,D)
                    bb = ntuple(_ -> nextfloat(x),D)
                    oo = ntuple(_ -> x,D)
                    check_enclosure(oo,aa,bb)
                end
                # Relative subtraction and a naive midpoint would overflow.
                aa = ntuple(_ -> -floatmax(T),D)
                bb = ntuple(_ -> -floatmax(T)/T(2),D)
                oo = ntuple(_ -> floatmax(T),D)
                check_enclosure(oo,aa,bb)
                # A line parallel to an unbounded local axis still intersects.
                aa = ntuple(k -> k==1 ? -floatmax(T)/T(2) : zero(T),D)
                oo = ntuple(k -> k==1 ? floatmax(T)/T(2) : zero(T),D)
                vv = ntuple(k -> T(k==1),D)
                @test distance(oo,vv,aa,aa) == 0
            end
        end

        @testset "3D near-axis, subnormal direction and overflowing t" begin
            for T in (Float32,Float64)
                tiny = T===Float32 ? T(1e-30) : T(1e-200)
                for s in (tiny,nextfloat(zero(T))), sign in (-one(T),one(T))
                    o = T.((0,2,-2)); v = (sign,sign*s,sign*s)
                    a,b = T.((-1,-1,-1)),T.((1,1,1))
                    d2 = distance(o,v,a,b)
                    @test isfinite(d2)
                    @test isapprox(d2,T(2);rtol=32eps(T))
                    @test !prune(o,v,a,b,T(2),d2)
                    # Infinite-edge t overflows; the finite edge endpoint is valid.
                    p = T.((0,2,2)); e = T.((1,1,1)); vv = (one(T),s,s)
                    dd = LB._squared_distance_line_box_edge3d(p,vv,e,1,2,3,zero(T),zero(T))
                    @test isfinite(dd)
                    @test isapprox(dd,T(8);rtol=32eps(T))
                end
                # t itself overflows but t*v[i] and the closest edge point do not.
                h = floatmax(T)*T(0.9)
                v = T.((1,7,7)); v = v ./ sqrt(sum(abs2,v))
                p = (zero(T),h,h); e = (floatmax(T)/T(2),one(T),one(T))
                @test LB._squared_distance_line_box_edge3d(p,v,e,1,2,3,zero(T),zero(T)) == 0
            end
        end

        @testset "Tangency, degeneracy and extreme scales" begin
            for T in (Float32,Float64), D in (2,3,4)
                exponents = T===Float32 ? (-140,-70,0,70,120) : (-1000,-500,0,500,1000)
                for exponent in exponents
                    s = ldexp(one(T),exponent)
                    v = ntuple(k -> T(k==2),D)
                    # Full box, flattened box, and point box.
                    for shape in (:full,:flat,:point)
                        a = ntuple(k -> shape===:point || (shape===:flat && k==D) ? zero(T) : -s,D)
                        b = ntuple(k -> -a[k],D)
                        o = ntuple(k -> k==1 ? b[k]+s : zero(T),D)
                        check_enclosure(o,a,b)
                        ref = reference(o,v,a,b)
                        d2 = distance(o,v,a,b)
                        @test !isnan(d2)
                        r2 = nextfloat(T(ref)) # a radius known to include the true distance
                        @test !prune(o,v,a,b,r2,d2)
                        @test !prune(o,v,a,b,r2,floatmax(T)) # finite overestimate
                        if isfinite(d2) && isfinite(T(ref))
                            @test isapprox(d2,T(ref);rtol=64eps(T),atol=8nextfloat(zero(T)))
                        end
                        # Tangent to the original face, including zero-radius queries.
                        oo = ntuple(k -> k==1 ? b[k] : zero(T),D)
                        @test !prune(oo,v,a,b,zero(T),distance(oo,v,a,b))
                        @test !prune(oo,v,a,b,zero(T),one(T))
                    end
                end
                z = ntuple(_ -> zero(T),D)
                v = ntuple(k -> T(k==2),D)
                o = ntuple(k -> k==1 ? T(4) : zero(T),D)
                @test prune(o,v,z,z,one(T),T(16)) # certificate is not always false
                @test !prune(o,v,z,z,one(T),T(NaN))
                @test !prune(o,v,z,z,one(T),T(Inf))
                @test !prune(o,v,z,z,T(NaN),T(16))
                @test !prune(o,v,z,z,-one(T),T(16))
                bad = ntuple(k -> k==D ? T(NaN) : o[k],D)
                @test !prune(bad,v,z,z,one(T),T(16))
            end
        end

        @testset "Random high-precision oracle and safe pruning" begin
            rng = Xoshiro(0x524f554e44)
            for T in (Float32,Float64), D in (2,3,4), trial in 1:100
                s = T(10.0^rand(rng,-8:8))
                a = ntuple(_ -> s*randn(rng,T),D)
                b = ntuple(k -> a[k]+s*rand(rng,T),D)
                o = ntuple(_ -> s*randn(rng,T),D)
                v = ntuple(k -> trial%4==0 && k==1 ? zero(T) : randn(rng,T),D)
                v = v ./ sqrt(sum(abs2,v))
                check_enclosure(o,a,b)
                exact = reference(o,v,a,b)
                d2 = distance(o,v,a,b)
                scale2 = max(BigFloat(1),sum(max(abs(BigFloat(a[k])-o[k]),abs(BigFloat(b[k])-o[k]))^2 for k in 1:D))
                @test isfinite(d2)
                @test abs(BigFloat(d2)-exact) <= 128eps(T)*scale2
                r2 = nextfloat(T(exact))
                @test !prune(o,v,a,b,r2,d2)
                @test !prune(o,v,a,b,r2,floatmax(T))
                # Also exercise rejecting radii and verify every rejection.
                r2 = T(exact/T(4))
                if prune(o,v,a,b,r2,d2)
                    @test exact > BigFloat(r2)
                end
            end
        end

        @testset "Two-leaf LBVH: no false-negative subtree pruning" begin
            for T in (Float32,Float64)
                x=T(1e8); y=nextfloat(x)
                enc=MortonEncoding(([x,y],zeros(T,2),zeros(T,2)))
                sort_by_morton!(enc)
                tree=LinearBVH(enc,ones(T,2))
                o=(y,zero(T),zero(T)); v=(zero(T),one(T),zero(T))
                node=Int32(1); hits=Int[]
                while node!=0
                    i=Int(node)
                    d2=LB._squared_distance_line_aabb(o,v,tree.aabb.min,tree.aabb.max,i)
                    if LB.is_leaf_id(node,2)
                        d2<=one(T) && push!(hits,LB.leaf_index(node,2))
                        node=tree.escape[i]
                    elseif LB._line_aabb_can_prune(o,v,tree.aabb.min,tree.aabb.max,i,one(T),d2)
                        node=tree.escape[i]
                    else
                        node=tree.left[i]
                    end
                end
                # Float64 spacing here is below radius=1, so both leaves hit.
                @test length(hits)==(T===Float32 ? 1 : 2)
            end
        end
    end
end
end # module

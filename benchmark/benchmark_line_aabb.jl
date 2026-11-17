# julia --project=. benchmark/benchmark_line_aabb.jl --n 100000 --repeats 10 --type Float64
include("line_aabb_common.jl")

function main()
    opts = options(["n"=>"100000", "repeats"=>"10", "type"=>"Float64", "seed"=>"20261118"])
    n, repeats = positive(opts["n"]), positive(opts["repeats"])
    T = float_type(opts["type"])
    rng = Xoshiro(parse(Int,opts["seed"]))
    println("# CPU D=3 infinite unit-direction lines; Julia $(VERSION); $T; seed=$(opts["seed"])")
    println("case,method,n,min_ns_per_query,median_ns_per_query,batch_bytes,checksum")
    for kind in ("random", "axis", "plane"), shape in ("box", "thin", "point", "intersect")
        origins, directions = queries(rng,T,n,kind)
        center = ntuple(_ -> rand(rng,T,n),3)
        extent = ntuple(d -> shape == "point" ? zeros(T,n) :
                            (shape == "thin" && d > 1 ? T(0.001) : T(0.2)).*rand(rng,T,n),3)
        lo = ntuple(d -> center[d].-extent[d],3)
        hi = ntuple(d -> center[d].+extent[d],3)
        if shape == "intersect"
            origins = [ntuple(d -> center[d][i],3) for i in 1:n]
        end
        # Validate all inputs, including exact zero direction components and point boxes.
        validate_distances(origins,directions,lo,hi)
        for (name,f) in METHODS
            s = measure(() -> distance_batch(f,origins,directions,lo,hi),repeats)
            @printf("%s_%s,%s,%d,%.3f,%.3f,%d,%.9g\n",kind,shape,name,n,
                    s.minimum*1e9/n,s.median*1e9/n,s.bytes,s.result)
        end
    end
end
main()

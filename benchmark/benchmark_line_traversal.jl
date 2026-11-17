# julia --project=. benchmark/benchmark_line_traversal.jl --nparts 10000,100000 --nlines 1000
include("line_aabb_common.jl")

function production_hits(tree,o,v,r,::Val{false})
    ids = Int[]
    LB.@LBVH_gather_line_traversal tree o v (r*r) leaf d2 begin
        push!(ids,leaf)
    end
    return sort!(ids)
end
function production_hits(tree,o,v,r,::Val{true})
    ids = Int[]
    LB.@LBVH_scatter_line_traversal tree o v r leaf d2 hb begin
        push!(ids,leaf)
    end
    return sort!(ids)
end

# Same stackless DFS, acceptance rule, and leaf callback for all three methods.
# Counters and leaf-ID collection compile out of the timed Val(false) variant.
function traverse(f::F, tree, o, v, radius, ::Val{Scatter}, ::Val{Stats}, ids=nothing) where {F,Scatter,Stats}
    node = Int32(1)
    nleaf = LB.nleaf(tree)
    hits = visits = leaves = 0
    @inbounds while node != 0
        i = Int(node)
        Stats && (visits += 1)
        r = Scatter ? radius*tree.scale[i] : radius
        d2 = f(o,v,tree.aabb.min,tree.aabb.max,i)
        if LB.is_leaf_id(node,nleaf)
            Stats && (leaves += 1)
            if d2 <= r*r
                hits += 1
                Stats && ids !== nothing && push!(ids,LB.leaf_index(node,nleaf))
            end
            node = tree.escape[i]
        elseif d2 > r*r
            node = tree.escape[i]
        else
            node = tree.left[i]
        end
    end
    return Stats ? (hits,visits,leaves) : hits
end

function traversal_batch(f::F,tree,origins,directions,radius,mode) where {F}
    total = 0
    for i in eachindex(origins)
        total += traverse(f,tree,origins[i],directions[i],radius,mode,Val(false))
    end
    return total
end

function validate_traversal(tree,origins,directions,radius,mode::Val{Scatter}) where {Scatter}
    counts = zeros(Int,3,3)
    n = LB.nleaf(tree)
    for q in eachindex(origins)
        o,v = origins[q],directions[q]
        expected = Int[]
        # Point leaves: direct point-line reference, independent of tree pruning.
        for i in n:2n-1
            p = ntuple(d -> Float64(tree.aabb.min[d][i])-Float64(o[d]),3)
            vv = Float64.(v)
            t = sum(p .* vv)/sum(abs2,vv)
            d2 = sum(abs2,p .- t.*vv)
            r = Scatter ? radius*tree.scale[i] : radius
            d2 <= Float64(r*r) && push!(expected,i-n+1)
        end
        for (j,(_,f)) in enumerate(METHODS)
            ids = Int[]
            stats = traverse(f,tree,o,v,radius,mode,Val(true),ids)
            sort!(ids)
            if ids != expected
                leaf = first(symdiff(ids,expected))
                i = leaf+n-1
                rr = Scatter ? radius*tree.scale[i] : radius
                values = [(name,fn(o,v,tree.aabb.min,tree.aabb.max,i)) for (name,fn) in METHODS]
                ref = reference_distance(o,v,tree.aabb.min,tree.aabb.max,i)
                error("Leaf mismatch: method=$(METHODS[j][1]), query=$q, leaf=$leaf, " *
                      "radius2=$(rr*rr), reference_d2=$ref, distances=$values")
            end
            counts[j,:] .+= stats
        end
        @assert production_hits(tree,o,v,radius,mode) == expected "Production macro mismatch"
    end
    return counts
end

function main()
    opts = options(["nparts"=>"10000,100000", "nlines"=>"1000", "repeats"=>"10",
                    "type"=>"Float64", "seed"=>"20261118", "radius"=>"0.02", "validate"=>"32"])
    ns = positive.(split(opts["nparts"],','))
    nq,repeats = positive(opts["nlines"]),positive(opts["repeats"])
    nv = min(nq,positive(opts["validate"]))
    T = float_type(opts["type"])
    radius = parse(T,opts["radius"])
    isfinite(radius) && radius > 0 || error("--radius must be finite and positive")
    rng = Xoshiro(parse(Int,opts["seed"]))
    println("# CPU serial traversal D=3; Julia $(VERSION); $T; seed=$(opts["seed"])")
    println("# Build/sort excluded. Stats averaged over first $nv validated lines; timing over all $nq lines.")
    println("cloud,nparts,directions,mode,method,min_us_per_line,median_us_per_line,batch_bytes,hits,visited_per_line,leaves_per_line")
    for n in ns, cloud in ("uniform","clustered")
        coords = ntuple(_ -> rand(rng,T,n),3)
        if cloud == "clustered"
            for d in 1:3, i in 1:n
                coords[d][i] = T(0.25+0.5*isodd(i)) + T(0.06)*randn(rng,T)
            end
        end
        enc = MortonEncoding(coords)
        sort_by_morton!(enc)
        scale = radius .* (T(0.5) .+ rand(rng,T,n))
        tree = LinearBVH(enc,scale) # point leaves, matching interpolation trees
        for kind in ("random","axis","plane")
            origins,directions = queries(rng,T,nq,kind)
            for (label,mode,r) in (("gather",Val(false),radius),("scatter",Val(true),one(T)))
                counts = validate_traversal(tree,origins[1:nv],directions[1:nv],r,mode)
                totals = Int[]
                for (j,(name,f)) in enumerate(METHODS)
                    s = measure(() -> traversal_batch(f,tree,origins,directions,r,mode),repeats)
                    push!(totals,s.result)
                    @printf("%s,%d,%s,%s,%s,%.3f,%.3f,%d,%d,%.2f,%.2f\n",cloud,n,kind,label,name,
                            s.minimum*1e6/nq,s.median*1e6/nq,s.bytes,s.result,counts[j,2]/nv,counts[j,3]/nv)
                end
                @assert allequal(totals) "Batch hit counts differ"
            end
        end
    end
end
abspath(PROGRAM_FILE) == (@__FILE__) && main()

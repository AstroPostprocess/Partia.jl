# julia --project=. benchmark/benchmark_phantom_line.jl --files temp/st15co_00120,temp/disc_00102
include("benchmark_line_traversal.jl")
using ParticleIO
using LinearAlgebra

function phantom_clouds(file, ::Type{T}, sink_id) where {T}
    data = read_phantom(String(file); separate_types=:all)
    sinks = data[end]
    get(sinks.params,:itype,nothing) isa Integer && error("Expected a final sink table in $file")
    1 <= sink_id <= size(sinks.dfdata,1) || error("Invalid --sink index")
    center = ntuple(d -> T(sinks[sink_id,(:x,:y,:z)[d]]),3)
    clouds = map(data[1:end-1]) do p
        itype = get(p.params,:itype,nothing)
        itype isa Integer || error("Expected explicitly separated particle types in $file")
        label = itype == 1 ? "gas" : itype >= 7 ? "dust_type$(itype)" : "type$(itype)"
        coords = ntuple(d -> T.(p[!,(:x,:y,:z)[d]]) .- center[d],3)
        h = T.(p[!,:h])
        all(isfinite,h) && all(>(zero(T)),h) || error("Invalid smoothing lengths")
        all(c -> all(isfinite,c),coords) || error("Invalid coordinates")
        isempty(h) && error("Empty particle type $itype")
        (; label,coords,h)
    end
    return center,clouds
end

# PCA normal of the particle distribution. Deterministic subsample avoids a
# full N-by-3 temporary; normal/plane directions follow the actual disc tilt.
function disc_basis(coords)
    n = length(coords[1])
    idx = unique(round.(Int,range(1,n;length=min(n,10000))))
    x = hcat((Float64.(c[idx]) for c in coords)...)
    x .-= mean(x;dims=1)
    ev = eigen(Symmetric(transpose(x)*x))
    return Tuple(ev.vectors[:,1]),Tuple(ev.vectors[:,3])
end

function cloud_queries(rng,coords,h,nq,kind,normal,tangent)
    T = eltype(h)
    origins = Vector{NTuple{3,T}}(undef,nq)
    directions = similar(origins)
    for q in 1:nq
        i = rand(rng,eachindex(h))
        # Sample the occupied region, then jitter by one local smoothing length.
        origins[q] = ntuple(d -> coords[d][i]+h[i]*(T(2)*rand(rng,T)-one(T)),3)
        v = kind == "face_on" ? T.(normal) : kind == "edge_on" ? T.(tangent) : ntuple(_ -> randn(rng,T),3)
        directions[q] = v ./ sqrt(sum(abs2,v))
    end
    return origins,directions
end

function benchmark_phantom_cloud(rng,file,cloud,::Type{T},nq,nmicro,nv,repeats,kvalid) where {T}
    (;label,coords,h) = cloud
    n = length(h)
    radii = sqrt.(coords[1].^2 .+ coords[2].^2 .+ coords[3].^2)
    @printf("# %s %s: N=%d; M1-relative r[p50,p90,p99]=%s; h[p10,p50,p90]=%s\n",
            basename(file),label,n,string(quantile(radii,[0.5,0.9,0.99])),string(quantile(h,[0.1,0.5,0.9])))
    normal,tangent = disc_basis(coords)
    println("# PCA face-on direction=$normal; edge-on direction=$tangent")
    enc = MortonEncoding(coords)
    sort_by_morton!(enc)
    sorted_h = h[enc.order]
    tree = LinearBVH(enc,sorted_h)
    gather_radius = kvalid*median(h)
    println("# gather radius=$gather_radius; scatter Kvalid=$kvalid with actual per-particle h")
    for kind in ("random","face_on","edge_on")
        origins,directions = cloud_queries(rng,coords,h,nq,kind,normal,tangent)
        # Uniformly sample actual internal-node AABBs. This isolates the distance
        # calculation on tree geometry, not the traversal's visitation distribution.
        nodes = rand(rng,1:max(n-1,1),nmicro)
        lo = ntuple(d -> tree.aabb.min[d][nodes],3)
        hi = ntuple(d -> tree.aabb.max[d][nodes],3)
        oi = [origins[mod1(i,nq)] for i in 1:nmicro]
        vi = [directions[mod1(i,nq)] for i in 1:nmicro]
        validate_distances(oi,vi,lo,hi)
        for (name,f) in METHODS
            s = measure(() -> distance_batch(f,oi,vi,lo,hi),repeats)
            @printf("distance,%s,%s,%d,%s,internal,%s,%d,%.6f,%.6f,%d,%.12g,NaN,NaN\n",
                    basename(file),label,n,kind,name,nmicro,s.minimum*1e6/nmicro,s.median*1e6/nmicro,s.bytes,s.result)
        end
        for (mode_name,mode,r) in (("gather",Val(false),gather_radius),("scatter",Val(true),kvalid))
            counts = validate_traversal(tree,origins[1:nv],directions[1:nv],r,mode)
            totals = Int[]
            for (j,(name,f)) in enumerate(METHODS)
                s = measure(() -> traversal_batch(f,tree,origins,directions,r,mode),repeats)
                push!(totals,s.result)
                @printf("traversal,%s,%s,%d,%s,%s,%s,%d,%.6f,%.6f,%d,%d,%.3f,%.3f\n",
                        basename(file),label,n,kind,mode_name,name,nq,s.minimum*1e6/nq,s.median*1e6/nq,
                        s.bytes,s.result,counts[j,2]/nv,counts[j,3]/nv)
            end
            @assert allequal(totals) "Batch hit counts differ"
        end
        flush(stdout)
    end
end

function phantom_main()
    opts = options(["files"=>"temp/st15co_00000,temp/st15co_00120,temp/disc_00102",
                    "type"=>"Float64", "nlines"=>"256", "nmicro"=>"10000", "validate"=>"8",
                    "repeats"=>"5", "seed"=>"20261118", "sink"=>"1", "kvalid"=>"3"])
    T = float_type(opts["type"])
    nq,nmicro,repeats = positive(opts["nlines"]),positive(opts["nmicro"]),positive(opts["repeats"])
    nv = min(nq,positive(opts["validate"]))
    sink = positive(opts["sink"])
    kvalid = parse(T,opts["kvalid"])
    isfinite(kvalid) && kvalid > 0 || error("--kvalid must be finite and positive")
    rng = Xoshiro(parse(Int,opts["seed"]))
    println("# CPU serial D=3; Julia=$VERSION; type=$T; seed=$(opts["seed"]); repeats=$repeats")
    println("# CPU=$(Sys.CPU_NAME); Julia threads=$(Threads.nthreads()); traversal uses one thread")
    println("# Reading/building/validation excluded; full active populations; node stats on first $nv lines")
    println("stage,file,population,nparts,directions,mode,method,nqueries,min_us_per_query,median_us_per_query,batch_bytes,checksum_or_hits,visited_per_line,leaves_per_line")
    flush(stdout)
    for file in split(opts["files"],',')
        isfile(file) || error("Missing Phantom dump: $file")
        center,clouds = phantom_clouds(file,T,sink)
        println("# $(basename(file)): sink[$sink] position=$center; coordinates shifted to this origin")
        flush(stdout)
        for cloud in clouds
            benchmark_phantom_cloud(rng,file,cloud,T,nq,nmicro,nv,repeats,kvalid)
        end
    end
end
abspath(PROGRAM_FILE) == (@__FILE__) && phantom_main()

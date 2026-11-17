using Partia, Random, Statistics, Printf
const LB = Partia.LinearBoundingVolumeHierarchy
include("eberly_line_aabb.jl")
const METHODS = (("lower_bound", LB._squared_distance_line_aabb_lower_bound),
                 ("exact_new", LB._squared_distance_line_aabb),
                 ("eberly_3d", eberly_line_aabb))

function options(defaults)
    opts = Dict(defaults)
    iseven(length(ARGS)) || error("Arguments must be --name value pairs")
    for i in 1:2:length(ARGS)
        key = replace(ARGS[i], r"^--" => "")
        haskey(opts, key) || error("Unknown option: $(ARGS[i])")
        opts[key] = ARGS[i+1]
    end
    return opts
end
positive(s) = (n = parse(Int, s); n > 0 || error("Expected positive integer"); n)
function float_type(s)
    s == "Float32" && return Float32
    s == "Float64" && return Float64
    error("--type must be Float32 or Float64")
end

function queries(rng, ::Type{T}, n, kind) where {T}
    origins = [ntuple(_ -> T(2)*rand(rng,T)-T(0.5), 3) for _ in 1:n]
    directions = map(1:n) do i
        v = ntuple(d -> kind == "axis" ? T(d == mod1(i,3)) :
                         kind == "plane" && d == mod1(i,3) ? zero(T) : randn(rng,T), 3)
        v ./ sqrt(sum(abs2,v))
    end
    return origins, directions
end

# Warm up every specialized closure; time batches and retain their result.
# Data generation, correctness checks, and tree building are outside timing.
function measure(f::F, repeats) where {F}
    result = f()
    times = Float64[]
    bytes = Int[]
    for _ in 1:repeats
        GC.gc()
        sample = @timed f()
        result = sample.value
        push!(times, sample.time)
        push!(bytes, sample.bytes)
    end
    return (; minimum=minimum(times), median=median(times), bytes=minimum(bytes), result)
end

function distance_batch(f::F, origins, directions, lo, hi) where {F}
    total = zero(eltype(first(origins)))
    @inbounds for i in eachindex(origins)
        total += f(origins[i], directions[i], lo, hi, i)
    end
    return total
end

# Independent, slower reference: minimize each quadratic piece between all
# line/slab breakpoints in Float64. Used only for validation, never timed.
function reference_distance(o, v, lo, hi, i)
    p, d = Float64.(o), Float64.(v)
    a = ntuple(k -> Float64(lo[k][i]), 3)
    b = ntuple(k -> Float64(hi[k][i]), 3)
    cuts = sort!([ (x-p[k])/d[k] for k in 1:3 if d[k] != 0 for x in (a[k],b[k]) ])
    edges = vcat(-Inf, cuts, Inf)
    best = Inf
    for j in 1:length(edges)-1
        l, h = edges[j], edges[j+1]
        t = isinf(l) ? h-max(1,abs(h)) : isinf(h) ? l+max(1,abs(l)) : (l+h)/2
        aa = bb = 0.0
        for k in 1:3
            x = p[k]+t*d[k]
            if x < a[k] || x > b[k]
                bound = x < a[k] ? a[k] : b[k]
                aa += d[k]^2
                bb += d[k]*(p[k]-bound)
            end
        end
        u = aa == 0 ? t : clamp(-bb/aa,l,h)
        best = min(best, sum(max(a[k]-p[k]-u*d[k], p[k]+u*d[k]-b[k],0.0)^2 for k in 1:3))
    end
    return best
end

function validate_distances(origins, directions, lo, hi)
    T = eltype(first(origins))
    for i in eachindex(origins)
        ref = reference_distance(origins[i], directions[i], lo, hi, i)
        # Eberly and the old point-line formula subtract squared quantities.
        # Their roundoff scales with the geometry, not only the final distance.
        scale2 = sum(max(abs(Float64(origins[i][k])-Float64(lo[k][i])),
                         abs(Float64(origins[i][k])-Float64(hi[k][i])))^2 for k in 1:3)
        tol = 128*eps(T)*max(1.0,ref,scale2)
        for (name,f) in METHODS
            d2 = f(origins[i], directions[i], lo, hi, i)
            @assert isfinite(d2) && d2 >= -tol (name,i,d2,ref)
            @assert name == "lower_bound" ? d2 <= ref+tol : abs(d2-ref) <= tol (name,i,d2,ref)
        end
    end
end

######################################################################################

# CPU/accelerator Morton encoding and LinearBVH scaling benchmark.
#
# Example:
#     julia benckmark/benchmark_gpu_cpu_lbvh.jl --backend auto \
#         --nparts 1000,10000,100000,1000000 --repeats 3 --dim 3 --type Float64

######################################################################################
using Random
using Printf
using Statistics
using ArgParse
using Partia
using UnsignedRadixSorts

function parse_commandline()
    settings = ArgParseSettings(
        description = "Compare CPU and automatically detected CUDA/Metal Morton + LBVH performance.",
    )
    @add_arg_table! settings begin
        "--backend"
            help = "accelerator backend: auto, CUDA, or Metal"
            arg_type = String
            default = "auto"
        "--nparts"
            help = "comma-separated particle counts"
            arg_type = String
            default = "1000,10000,100000,1000000"
        "--repeats"
            help = "number of measured samples after warm-up"
            arg_type = Int
            default = 6
        "--dim"
            help = "coordinate dimension (2 or 3)"
            arg_type = Int
            default = 3
        "--type"
            help = "floating-point type: Float32 or Float64; Metal always uses Float32"
            arg_type = String
            default = "Float64"
            dest_name = "float_type"
    end
    return parse_args(settings)
end

const OPTIONS = parse_commandline()
const REPEATS = OPTIONS["repeats"]
const DIM = OPTIONS["dim"]
const REQUESTED_TYPE = OPTIONS["float_type"]
const REQUESTED_BACKEND = lowercase(OPTIONS["backend"])
const NPARTS = try
    parse.(Int, split(OPTIONS["nparts"], ','))
catch
    error("--nparts must be a comma-separated list of integers")
end

DIM in (2, 3) || error("--dim must be 2 or 3")
REPEATS > 0 || error("--repeats must be positive")
all(>(0), NPARTS) || error("all particle counts must be positive")
REQUESTED_TYPE in ("Float32", "Float64") || error("--type must be Float32 or Float64")

"""Import an installed accelerator package and return it only when functional."""
function load_functional_backend(name :: Symbol)
    Base.find_package(String(name)) === nothing && return nothing
    backend = try
        Base.eval(@__MODULE__, :(import $(name)))
        Base.invokelatest(getfield, @__MODULE__, name)
    catch
        return nothing
    end
    Base.invokelatest(getproperty(backend, :functional)) || return nothing
    return backend
end

"""Select CUDA or Metal, preferring CUDA when auto-detection finds both."""
function select_backend()
    requested = REQUESTED_BACKEND
    requested in ("auto", "cuda", "metal") || error("--backend must be auto, CUDA, or Metal")

    if requested in ("auto", "cuda")
        backend = load_functional_backend(:CUDA)
        backend !== nothing && return (name=:CUDA, module_ref=backend)
        requested == "cuda" && error("CUDA.jl is not installed or CUDA.functional() is false")
    end
    if requested in ("auto", "metal") && Sys.isapple()
        backend = load_functional_backend(:Metal)
        backend !== nothing && return (name=:Metal, module_ref=backend)
    end
    requested == "metal" && !Sys.isapple() && error("Metal.jl requires macOS")
    error("no functional CUDA or Metal backend was detected")
end

# Load the selected package before defining benchmark functions, ensuring that
# its device APIs and Partia extension methods belong to their compilation world.
const SELECTED_BACKEND = select_backend()

"""
Return minimum, median, mean, and maximum runtime after one warm-up. The only
retained sample storage is an O(repeats) `Float64` timing vector.
"""
function sample_statistics(f, synchronize; repeats=REPEATS, prepare=() -> nothing)
    prepare()
    synchronize()
    result = f()
    synchronize()
    result = nothing
    GC.gc()

    times = Vector{Float64}(undef, repeats)
    for sample in 1:repeats
        GC.gc()
        prepare()
        synchronize()
        start = time_ns()
        result = f()
        synchronize()
        times[sample] = (time_ns() - start) * 1.0e-9
        result = nothing
    end
    return (
        minimum = minimum(times),
        median = median(times),
        mean = mean(times),
        maximum = maximum(times),
    )
end

function print_rule()
    println("├─────────────┼─────────┼────────────┼────────────┼────────────┼────────────┼────────────┼──────────┤")
end

"""Print a complete titled table header for one particle-count block."""
function print_table_header()
    println("┌─────────────┬─────────┬────────────┬────────────┬────────────┬────────────┬────────────┬──────────┐")
    println("│ Stage       │ Backend │      Npart │    Minimum │     Median │       Mean │    Maximum │  Speedup │")
    println("│             │         │            │       (ms) │       (ms) │       (ms) │       (ms) │  by mean │")
    print_rule()
end

function print_rows(stage, n, accelerator_name, cpu, accelerator)
    speedup = cpu.mean / accelerator.mean
    @printf("│ %-11s │ %-7s │ %10d │ %10.3f │ %10.3f │ %10.3f │ %10.3f │ %7s  │\n",
            stage, "CPU", n, 1e3cpu.minimum, 1e3cpu.median,
            1e3cpu.mean, 1e3cpu.maximum, "1.00x")
    @printf("│ %-11s │ %-7s │ %10d │ %10.3f │ %10.3f │ %10.3f │ %10.3f │ %7.2fx │\n",
            "", string(accelerator_name), n, 1e3accelerator.minimum,
            1e3accelerator.median, 1e3accelerator.mean,
            1e3accelerator.maximum, speedup)
end

function benchmark_size(n, config)
    size_started = time_ns()
    TF = config.float_type

    # Generate and transfer one particle count at a time.
    rng = MersenneTwister(0xB16B00B5 + n)
    host_coords = ntuple(_ -> rand(rng, TF, n), DIM)
    host_scale = rand(rng, TF, n) .* TF(0.2) .+ TF(0.01)
    device_coords = map(config.to_device, host_coords)
    device_scale_source = config.to_device(host_scale)

    cpu_sort_workspace = OnesweepWorkspace(Vector{UInt64})
    device_sort_workspace = OnesweepWorkspace(config.device_vector_type{UInt64})

    # Compare the allocating public constructors directly. The standard path
    # owns a coordinate copy, whereas MortonEncoding! sorts caller-owned
    # coordinates in place and therefore omits that copy.
    cpu_standard_encode() = MortonEncoding(host_coords; SortWorkSpace=cpu_sort_workspace)
    device_standard_encode() = MortonEncoding(device_coords; SortWorkSpace=device_sort_workspace)

    cpu_no_copy_coords = map(similar, host_coords)
    device_no_copy_coords = map(similar, device_coords)
    function prepare_cpu_no_copy()
        foreach(copyto!, cpu_no_copy_coords, host_coords)
        return nothing
    end
    function prepare_device_no_copy()
        foreach(copyto!, device_no_copy_coords, device_coords)
        return nothing
    end
    cpu_no_copy_encode() = DIM == 2 ?
        MortonEncoding!(cpu_no_copy_coords[1], cpu_no_copy_coords[2]; SortWorkSpace=cpu_sort_workspace) :
        MortonEncoding!(cpu_no_copy_coords[1], cpu_no_copy_coords[2], cpu_no_copy_coords[3]; SortWorkSpace=cpu_sort_workspace)
    device_no_copy_encode() = DIM == 2 ?
        MortonEncoding!(device_no_copy_coords[1], device_no_copy_coords[2]; SortWorkSpace=device_sort_workspace) :
        MortonEncoding!(device_no_copy_coords[1], device_no_copy_coords[2], device_no_copy_coords[3]; SortWorkSpace=device_sort_workspace)

    cpu_standard_statistics = sample_statistics(cpu_standard_encode, () -> nothing)
    device_standard_statistics = sample_statistics(device_standard_encode, config.synchronize)
    print_rows("enc-copy", n, config.name, cpu_standard_statistics, device_standard_statistics)

    # Restore caller-owned coordinates outside the timed region, so this row
    # measures MortonEncoding! rather than the benchmark reset operation.
    cpu_no_copy_statistics = sample_statistics(cpu_no_copy_encode, () -> nothing; prepare=prepare_cpu_no_copy)
    device_no_copy_statistics = sample_statistics(device_no_copy_encode, config.synchronize; prepare=prepare_device_no_copy)
    print_rows("enc-no-copy", n, config.name, cpu_no_copy_statistics, device_no_copy_statistics)

    # Allocate encoding storage once through the no-copy API. Separate working
    # coordinates are necessary because repeats still need the unsorted inputs.
    cpu_encoding_coords = map(copy, host_coords)
    device_encoding_coords = map(copy, device_coords)
    cpu_enc = DIM == 2 ?
        MortonEncoding!(cpu_encoding_coords[1], cpu_encoding_coords[2]; SortWorkSpace=cpu_sort_workspace) :
        MortonEncoding!(cpu_encoding_coords[1], cpu_encoding_coords[2], cpu_encoding_coords[3]; SortWorkSpace=cpu_sort_workspace)
    device_enc = DIM == 2 ?
        MortonEncoding!(device_encoding_coords[1], device_encoding_coords[2]; SortWorkSpace=device_sort_workspace) :
        MortonEncoding!(device_encoding_coords[1], device_encoding_coords[2], device_encoding_coords[3]; SortWorkSpace=device_sort_workspace)
    cpu_encode() = DIM == 2 ?
        build!(cpu_enc, host_coords[1], host_coords[2], cpu_sort_workspace) :
        build!(cpu_enc, host_coords[1], host_coords[2], host_coords[3], cpu_sort_workspace)
    device_encode() = DIM == 2 ?
        build!(device_enc, device_coords[1], device_coords[2], device_sort_workspace) :
        build!(device_enc, device_coords[1], device_coords[2], device_coords[3], device_sort_workspace)

    cpu_enc_statistics = sample_statistics(cpu_encode, () -> nothing)
    device_enc_statistics = sample_statistics(device_encode, config.synchronize)
    print_rows("enc-reuse", n, config.name, cpu_enc_statistics, device_enc_statistics)

    # Reuse the sorted encoding and scale vectors for isolated LBVH timing.
    cpu_encode()
    device_encode()
    config.synchronize()
    cpu_scale = copy(host_scale)
    Base.permute!(cpu_scale, cpu_enc.order)
    device_scale = copy(device_scale_source)
    Base.permute!(device_scale, device_enc.order)

    # Measure the standard allocating constructor separately from build! so the
    # cost of creating hierarchy storage remains visible in the comparison.
    cpu_standard_build() = LinearBVH(cpu_enc, cpu_scale)
    device_standard_build() = LinearBVH(device_enc, device_scale)
    cpu_standard_bvh_statistics = sample_statistics(cpu_standard_build, () -> nothing)
    device_standard_bvh_statistics = sample_statistics(device_standard_build, config.synchronize)
    print_rows("lbvh-new", n, config.name, cpu_standard_bvh_statistics, device_standard_bvh_statistics)

    # Allocate hierarchy and rendezvous storage once; isolated reusable timing
    # measures build! without allocating O(Npart) output storage per repeat.
    cpu_lbvh = LinearBVH(cpu_enc, cpu_scale)
    device_lbvh = LinearBVH(device_enc, device_scale)
    cpu_store = Vector{Int32}(undef, n - 1)
    device_store = config.device_vector_type{Int32}(undef, n - 1)
    cpu_build() = build!(cpu_lbvh, cpu_store, cpu_enc, cpu_scale)
    device_build() = build!(device_lbvh, device_store, device_enc, device_scale)
    cpu_bvh_statistics = sample_statistics(cpu_build, () -> nothing)
    device_bvh_statistics = sample_statistics(device_build, config.synchronize)
    print_rows("lbvh-reuse", n, config.name, cpu_bvh_statistics, device_bvh_statistics)

    # End-to-end timing includes reusable encoding, scale reordering, and
    # reusable LBVH building without reconstructing O(Npart) result storage.
    # Reuse scale scratch vectors so repeats do not allocate another O(Npart)
    # host and device vector merely to apply the Morton permutation.
    cpu_total_scale = similar(host_scale)
    device_total_scale = similar(device_scale_source)
    function cpu_total()
        enc = cpu_encode()
        copyto!(cpu_total_scale, host_scale)
        Base.permute!(cpu_total_scale, enc.order)
        return build!(cpu_lbvh, cpu_store, enc, cpu_total_scale)
    end
    function device_total()
        enc = device_encode()
        copyto!(device_total_scale, device_scale_source)
        Base.permute!(device_total_scale, enc.order)
        return build!(device_lbvh, device_store, enc, device_total_scale)
    end
    cpu_total_statistics = sample_statistics(cpu_total, () -> nothing)
    device_total_statistics = sample_statistics(device_total, config.synchronize)
    print_rows("total", n, config.name, cpu_total_statistics, device_total_statistics)

    # Release this size before allocating the next one.
    cpu_enc = device_enc = cpu_scale = device_scale = nothing
    cpu_lbvh = device_lbvh = cpu_store = device_store = nothing
    cpu_total_scale = device_total_scale = nothing
    host_coords = host_scale = device_coords = device_scale_source = nothing
    cpu_encoding_coords = device_encoding_coords = nothing
    cpu_no_copy_coords = device_no_copy_coords = nothing
    cpu_sort_workspace = device_sort_workspace = nothing
    GC.gc(true)
    config.reclaim()

    elapsed = (time_ns() - size_started) * 1.0e-9
    println("└─────────────┴─────────┴────────────┴────────────┴────────────┴────────────┴────────────┴──────────┘")
    @printf("Npart=%d wall time: %.2f s\n\n", n, elapsed)
    return elapsed
end

function main()
    selected = SELECTED_BACKEND
    backend = selected.module_ref
    is_metal = selected.name === :Metal
    TF = is_metal ? Float32 : (REQUESTED_TYPE == "Float32" ? Float32 : Float64)
    REQUESTED_TYPE == "Float64" && is_metal && @info "Metal benchmark uses Float32 by backend design"

    vector_name = is_metal ? :MtlVector : :CuVector
    to_device_name = is_metal ? :mtl : :cu
    reclaim = isdefined(backend, :reclaim) ? getproperty(backend, :reclaim) : (() -> nothing)
    device_description = is_metal ? string(getproperty(backend, :device)()) :
        getproperty(backend, :name)(getproperty(backend, :device)())
    config = (
        name = selected.name,
        float_type = TF,
        device_vector_type = getproperty(backend, vector_name),
        to_device = getproperty(backend, to_device_name),
        synchronize = getproperty(backend, :synchronize),
        reclaim = reclaim,
    )

    println()
    println("Partia Morton encoding + LinearBVH benchmark")
    println("Backend : $(selected.name) ($(device_description))")
    println("Config  : D=$DIM, type=$TF, repeats=$REPEATS, CPU threads=$(Threads.nthreads())")
    println("Npart   : $(join(NPARTS, ", "))")
    println()
    total_started = time_ns()
    measured_wall = 0.0
    for n in NPARTS
        print_table_header()
        measured_wall += benchmark_size(n, config)
    end
    total_elapsed = (time_ns() - total_started) * 1.0e-9
    @printf("Benchmark wall time: %.2f s (sum by Npart: %.2f s)\n", total_elapsed, measured_wall)
end

main()

######################################################################################

#  Test: Optional CUDA and Metal Backends
#  What this file tests
#  Detects optional GPU packages and runs the shared accelerator regression suite:
#  1. Package detection
#     • Searches the active load path and the default Julia environment.
#     • Does not install CUDA.jl or Metal.jl as part of the test run.
#  2. Runtime capability detection
#     • Loads an installed package and checks `functional()` before allocating
#       device memory or compiling kernels.
#     • Records one skipped test when the package or functional device is absent.
#  3. Backend parity
#     • Calls the same test body for CUDA and Metal, using Float32 CPU results as
#       the common reference.

######################################################################################
using Test
using Partia

# ========================== Backend startup state =========================== #

if !isdefined(@__MODULE__, :cuda_status)
    include("accelerator_test_setup.jl")
end

# ========================== Shared includes ================================= #

@static if !isdefined(@__MODULE__, :run_accelerator_test_suite)
    include("accelerator_test_common.jl")
end

function accelerator_test_unavailable(name, status)
    if status.state === :skip
        @info "$name tests skipped" reason = status.reason
        @test_skip false
    else
        @error "$name backend could not be tested" reason = status.reason
        @test false
    end
    return nothing
end

# ============================== Test body =================================== #

@testset "Accelerators -- optional backends" begin
    @testset "Accelerator -- CUDA backend" begin
        if cuda_status.state !== :ready
            accelerator_test_unavailable("CUDA", cuda_status)
        else
            CUDA_mod = cuda_status.module_ref
            config = (
                name = "CUDA",
                to_device = Partia.to_CuVector,
                to_device_vector = getproperty(CUDA_mod, :cu),
                to_host = Partia.to_HostVector,
                synchronize = getproperty(CUDA_mod, :synchronize),
            )
            Base.invokelatest(run_accelerator_test_suite, config)
        end
    end

    @testset "Accelerator -- Metal backend" begin
        if metal_status.state !== :ready
            accelerator_test_unavailable("Metal", metal_status)
        else
            Metal_mod = metal_status.module_ref
            config = (
                name = "Metal",
                to_device = Partia.to_MtlVector,
                to_device_vector = getproperty(Metal_mod, :mtl),
                to_host = Partia.to_HostVector,
                synchronize = getproperty(Metal_mod, :synchronize),
            )
            Base.invokelatest(run_accelerator_test_suite, config)
        end
    end
end

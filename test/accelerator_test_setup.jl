######################################################################################

# Optional accelerator package detection and startup loading.

######################################################################################
using Partia

function accelerator_test_find_package(name :: String)
    path = Base.find_package(name)
    path !== nothing && return path

    # Pkg.test() may isolate LOAD_PATH from the user's default environment.
    default_environment = "@v#.#"
    if default_environment ∉ LOAD_PATH
        push!(LOAD_PATH, default_environment)
        path = Base.find_package(name)
    end
    return path
end

function accelerator_test_load_package(name :: Symbol)
    accelerator_test_find_package(String(name)) === nothing && return (
        state = :skip,
        module_ref = nothing,
        reason = "$(name).jl is not installed in the active or default Julia environment.",
    )

    module_ref = try
        Base.eval(@__MODULE__, :(import $(name)))
        Base.invokelatest(getfield, @__MODULE__, name)
    catch err
        return (
            state = :error,
            module_ref = nothing,
            reason = "$(name).jl was found but could not be imported: $(sprint(showerror, err))",
        )
    end

    functional = try
        Base.invokelatest(getproperty(module_ref, :functional))
    catch err
        return (
            state = :skip,
            module_ref = nothing,
            reason = "$(name).functional() could not initialise a usable device: $(sprint(showerror, err))",
        )
    end
    functional || return (
        state = :skip,
        module_ref = nothing,
        reason = "$(name).jl is installed, but $(name).functional() returned false.",
    )

    extension_name = Symbol(name, :Ext)
    extension = Base.invokelatest(Base.get_extension, Partia, extension_name)
    extension === nothing && return (
        state = :error,
        module_ref = nothing,
        reason = "Partia.$extension_name was not loaded after importing $(name).jl.",
    )

    return (state = :ready, module_ref = module_ref, reason = nothing)
end

# Import usable backends at test startup, but defer their test bodies until the
# final accelerator section of runtests.jl.
cuda_status = accelerator_test_load_package(:CUDA)
metal_status = Sys.isapple() ? accelerator_test_load_package(:Metal) : (
    state = :skip,
    module_ref = nothing,
    reason = "Metal.jl requires macOS.",
)

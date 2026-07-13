module Tools
using Metal

# Atomic operations
include(joinpath(@__DIR__, "atomic_compare_exchange.jl"))

end

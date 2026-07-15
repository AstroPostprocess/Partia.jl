module Tools
using Metal
import Partia

# Atomic operations
include(joinpath(@__DIR__, "atomic_compare_exchange.jl"))

# Check sorted
include(joinpath(@__DIR__, "issorted.jl"))

end

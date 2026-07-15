######################################################################################

#  Partia.jl - Test Suite Entry Point
#  Run with:  julia test/runtests.jl
#             or: include("test/runtests.jl") from the REPL
#  Ordering convention
#  1. Equation of state            (thermodynamic helpers)
#  2. Utility functions            (Tools: coordinates, arrays)
#  3. Spatial data structures      (BRT, LBVH)
#  4. Kernel functions             (M4/M5/M6, Wendland C2/C4/C6)
#  5. Interpolation infrastructure (constructors, traversal, physics)
#  6. Additional core checks       (type stability, traversal)
#  7. Optional accelerator checks  (CUDA and Metal, when functional)

######################################################################################
using Test
using Partia

# ======================= Accelerator startup detection ====================== #

# Detect and load optional GPU backends now, while deferring their test bodies
# until the final section of the suite.
include("accelerator_test_setup.jl")

# ========================== Equation of state =============================== #

include("eos_tests.jl")

# ========================== Utility functions =============================== #

include("tools_tests.jl")
include("coordinate_tests.jl")
include("frame_tests.jl")

# ========================== Spatial data structures ========================= #

include("neighbor_search_tests.jl")

# ========================== Kernel functions ================================ #

include("kernel_function_tests.jl")

# ========================== Interpolation =================================== #

include("interpolation_test_common.jl")
include("interpolation_analytic_test_common.jl")
include("interpolation_tests.jl")
include("grid_interpolation_test_common.jl")
include("point_samples_interpolation_tests.jl")
include("structured_grid_interpolation_tests.jl")
include("traversal_analytic.jl")
include("type_stability.jl")

# ========================== Optional accelerators =========================== #

include("accelerator_tests.jl")

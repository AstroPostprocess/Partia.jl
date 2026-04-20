
"""
EOS

Equation-of-state types and thermodynamic property evaluators.
"""
module EOS

# Abstract type
include(joinpath(@__DIR__, "AbstractType", "AbstractEOS.jl"))
include(joinpath(@__DIR__, "AbstractType", "AbstractUnit.jl"))

# Adiabatic
include(joinpath(@__DIR__, "Adiabatic", "SoundSpeed.jl"))
include(joinpath(@__DIR__, "Adiabatic", "Pressure.jl"))
include(joinpath(@__DIR__, "Adiabatic", "Temperature.jl"))
include(joinpath(@__DIR__, "Adiabatic", "Enthalpy.jl"))

# Isothermal
include(joinpath(@__DIR__, "Isothermal", "SoundSpeed.jl"))
include(joinpath(@__DIR__, "Isothermal", "Pressure.jl"))
include(joinpath(@__DIR__, "Isothermal", "Temperature.jl"))
include(joinpath(@__DIR__, "Isothermal", "Enthalpy.jl"))

# Locally isothermal
include(joinpath(@__DIR__, "LocallyIsothermal", "SoundSpeed.jl"))
include(joinpath(@__DIR__, "LocallyIsothermal", "Pressure.jl"))
include(joinpath(@__DIR__, "LocallyIsothermal", "Temperature.jl"))
include(joinpath(@__DIR__, "LocallyIsothermal", "Enthalpy.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end

end

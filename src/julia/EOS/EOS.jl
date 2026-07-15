
"""
EOS

Equation-of-state types and thermodynamic property evaluators.
"""
module EOS

# Abstract type
include(joinpath(@__DIR__, "AbstractType", "AbstractEOS.jl"))
include(joinpath(@__DIR__, "AbstractType", "AbstractUnit.jl"))

# Adiabatic
include(joinpath(@__DIR__, "Adiabatic", "sound_speed.jl"))
include(joinpath(@__DIR__, "Adiabatic", "pressure.jl"))
include(joinpath(@__DIR__, "Adiabatic", "temperature.jl"))
include(joinpath(@__DIR__, "Adiabatic", "enthalpy.jl"))

# Isothermal
include(joinpath(@__DIR__, "Isothermal", "sound_speed.jl"))
include(joinpath(@__DIR__, "Isothermal", "pressure.jl"))
include(joinpath(@__DIR__, "Isothermal", "temperature.jl"))
include(joinpath(@__DIR__, "Isothermal", "enthalpy.jl"))

# Locally isothermal
include(joinpath(@__DIR__, "LocallyIsothermal", "sound_speed.jl"))
include(joinpath(@__DIR__, "LocallyIsothermal", "pressure.jl"))
include(joinpath(@__DIR__, "LocallyIsothermal", "temperature.jl"))
include(joinpath(@__DIR__, "LocallyIsothermal", "enthalpy.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end

end

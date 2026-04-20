# EOS Types
abstract type AbstractEOS end
struct Adiabatic <: AbstractEOS end
struct Isothermal <: AbstractEOS end
struct LocallyIsothermal <: AbstractEOS end
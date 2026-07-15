######################################################################################

# EOS Types

######################################################################################
"""Abstract supertype for equation-of-state dispatch tags."""
abstract type AbstractEOS end

"""Dispatch tag selecting an adiabatic equation of state."""
struct Adiabatic <: AbstractEOS end

"""Dispatch tag selecting an isothermal equation of state."""
struct Isothermal <: AbstractEOS end

"""Dispatch tag selecting a locally isothermal equation of state."""
struct LocallyIsothermal <: AbstractEOS end

"""
    Enthalpy( :: Type{Adiabatic}, u :: T, γ :: T) :: T where {T <: AbstractFloat}

Compute the specific enthalpy for an adiabatic ideal gas from the specific
internal energy `u` and adiabatic index `γ`, using

    h = γ u.

Returns `NaN` if `u < 0` or `γ < 1`.

# Parameters
- ` :: Type{Adiabatic}` : Dispatch tag indicating the adiabatic ideal-gas EOS.
- `u :: T` : Specific internal energy.
- `γ :: T` : Adiabatic index.

# Returns
- `T` : The computed specific enthalpy, or `NaN` if input is unphysical.
"""
@inline function Enthalpy( :: Type{Adiabatic}, u :: T, γ :: T) :: T where {T <: AbstractFloat}
    return (u < zero(T) || γ < one(T)) ? T(NaN) : γ * u
end

"""
    Enthalpy( :: Type{Adiabatic}, u :: AbstractFloat, γ :: AbstractFloat)

Compute the specific enthalpy for an adiabatic ideal gas from the specific
internal energy `u` and adiabatic index `γ`, after promoting both inputs to a
common floating-point type.

The enthalpy is evaluated as

    h = γ u.

Returns `NaN` if `u < 0` or `γ < 1` after promotion.

# Parameters
- ` :: Type{Adiabatic}` : Dispatch tag indicating the adiabatic ideal-gas EOS.
- `u :: AbstractFloat` : Specific internal energy.
- `γ :: AbstractFloat` : Adiabatic index.

# Returns
- `promote_type(typeof(u), typeof(γ))` : The computed specific enthalpy in the promoted floating-point type, or `NaN` if input is unphysical.
"""
@inline function Enthalpy( :: Type{Adiabatic}, u :: AbstractFloat, γ :: AbstractFloat)
    up, γp = promote(u, γ)
    return Enthalpy(Adiabatic, up, γp)
end
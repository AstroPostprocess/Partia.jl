"""
    Enthalpy(::Type{Isothermal}, u::T, cs::T) :: T where {T<:AbstractFloat}

Compute the specific enthalpy for an isothermal gas from the specific internal
energy `u` and isothermal sound speed `cs`, using

    h = u + c_s^2.

This follows from the definition

    h = u + P / ρ,

together with the isothermal equation of state

    P = ρ c_s^2.

Returns `NaN` if `u < 0` or `cs < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating the isothermal EOS.
- `u::T` : Specific internal energy.
- `cs::T` : Constant isothermal sound speed.

# Returns
- `T` : The computed specific enthalpy, or `NaN` if input is unphysical.
"""
@inline function Enthalpy(::Type{Isothermal}, u::T, cs::T) :: T where {T<:AbstractFloat}
    return (u < zero(T) || cs < zero(T)) ? T(NaN) : u + cs * cs
end

"""
    Enthalpy(::Type{Isothermal}, u::AbstractFloat, cs::AbstractFloat)

Compute the specific enthalpy for an isothermal gas from the specific internal
energy `u` and isothermal sound speed `cs`, after promoting both inputs to a
common floating-point type.

The enthalpy is evaluated as

    h = u + c_s^2.

Returns `NaN` if `u < 0` or `cs < 0` after promotion.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating the isothermal EOS.
- `u::AbstractFloat` : Specific internal energy.
- `cs::AbstractFloat` : Constant isothermal sound speed.

# Returns
- `promote_type(typeof(u), typeof(cs))` : The computed specific enthalpy in the
  promoted floating-point type, or `NaN` if input is unphysical.
"""
@inline function Enthalpy(::Type{Isothermal}, u::AbstractFloat, cs::AbstractFloat)
    up, csp = promote(u, cs)
    return Enthalpy(Isothermal, up, csp)
end
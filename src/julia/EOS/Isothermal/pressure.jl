"""
    Pressure( :: Type{Isothermal}, ρ :: T, cs :: T) :: T where {T <: AbstractFloat}

Compute the isothermal gas pressure using the equation of state

    P = ρ c_s²,

where `ρ` is the mass density and `c_s` is the isothermal sound speed.
Returns `NaN` if `ρ < 0`.

# Parameters
- ` :: Type{Isothermal}` : Dispatch tag indicating isothermal pressure calculation.
- `ρ :: T` : Mass density.
- `cs :: T` : Constant isothermal sound speed.

# Returns
- `T` : The computed pressure, or `NaN` if input is unphysical.
"""
@inline function Pressure( :: Type{Isothermal}, ρ :: T, cs :: T) :: T where {T <: AbstractFloat}
    if ρ < zero(T)
        return T(NaN)
    end
    return ρ * cs^2
end

"""
    Pressure( :: Type{Isothermal}, ρ :: AbstractFloat, cs :: AbstractFloat)

Compute the isothermal gas pressure with automatic type promotion, using the equation of state

    P = ρ c_s²,

where `ρ` is the mass density and `c_s` is the isothermal sound speed.
Returns `NaN` if `ρ < 0`.

# Parameters
- ` :: Type{Isothermal}` : Dispatch tag indicating isothermal pressure calculation.
- `ρ :: AbstractFloat` : Mass density.
- `cs :: AbstractFloat` : Constant isothermal sound speed.

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the
  computed pressure, or `NaN` if input is unphysical.
"""
@inline function Pressure( :: Type{Isothermal}, ρ :: AbstractFloat, cs :: AbstractFloat)
    ρp, csp = promote(ρ, cs)
    T = typeof(ρp)
    if ρp < zero(T)
        return T(NaN)
    end
    return ρp * csp^2
end

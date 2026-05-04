"""
    Pressure( :: Type{LocallyIsothermal}, ρ :: T, r :: T, cs0 :: T, q :: T) :: T where {T <: AbstractFloat}

Compute the locally isothermal gas pressure profile, defined as

    P(r) = ρ [ cs0 * r^(-q) ]²,

where `ρ` is the mass density, `cs0` is a reference sound speed, and `q` is the radial power-law index.
Returns `NaN` if `ρ < 0`, `r ≤ 0`, or `cs0 < 0`.

# Parameters
- ` :: Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal pressure calculation.
- `ρ :: T` : Mass density.
- `r :: T` : Radial position.
- `cs0 :: T` : Reference sound speed at `r = 1`.
- `q :: T` : Power-law exponent controlling radial dependence.

# Returns
- `T` : The locally isothermal pressure at radius `r`, or `NaN` if input is unphysical.
"""
@inline function Pressure( :: Type{LocallyIsothermal}, ρ :: T, r :: T, cs0 :: T, q :: T) :: T where {T <: AbstractFloat}
    if ρ < zero(T) || r <= zero(T) || cs0 < zero(T)
        return T(NaN)
    end
    return ρ * (cs0 * r^(-q))^2
end

"""
    Pressure( :: Type{LocallyIsothermal}, ρ :: AbstractFloat, r :: AbstractFloat, cs0 :: AbstractFloat, q :: AbstractFloat)

Compute the locally isothermal gas pressure with automatic type promotion, defined as

    P(r) = ρ [ cs0 * r^(-q) ]²,

where `ρ` is the mass density, `cs0` is a reference sound speed, and `q` is the radial power-law index.
Returns `NaN` if `ρ < 0`, `r ≤ 0`, or `cs0 < 0`.

# Parameters
- ` :: Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal pressure calculation.
- `ρ :: AbstractFloat` : Mass density.
- `r :: AbstractFloat` : Radial position.
- `cs0 :: AbstractFloat` : Reference sound speed at `r = 1`.
- `q :: AbstractFloat` : Power-law exponent controlling radial dependence.

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the
  locally isothermal pressure at radius `r`, or `NaN` if input is unphysical.
"""
@inline function Pressure( :: Type{LocallyIsothermal}, ρ :: AbstractFloat, r :: AbstractFloat, cs0 :: AbstractFloat, q :: AbstractFloat)
    ρp, rp, cs0p, qp = promote(ρ, r, cs0, q)
    T = typeof(ρp)
    if ρp < zero(T) || rp <= zero(T) || cs0p < zero(T)
        return T(NaN)
    end
    return ρp * (cs0p * rp^(-qp))^2
end

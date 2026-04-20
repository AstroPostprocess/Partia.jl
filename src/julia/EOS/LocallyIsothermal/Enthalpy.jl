"""
    Enthalpy(::Type{LocallyIsothermal}, u::T, r::T, cs0::T, q::T) :: T where {T<:AbstractFloat}

Compute the specific enthalpy for a locally isothermal gas from the specific
internal energy `u`, radial position `r`, reference sound speed `cs0`, and
radial power-law index `q`, using

    h(r) = u + c_s(r)^2,

with

    c_s(r) = cs0 r^(-q),

so that

    h(r) = u + [ cs0 r^(-q) ]^2.

This follows from the definition

    h = u + P / ρ,

together with the locally isothermal equation of state

    P(r) = ρ [ cs0 r^(-q) ]^2.

Returns `NaN` if `u < 0`, `r ≤ 0`, or `cs0 < 0`.

# Parameters
- `::Type{LocallyIsothermal}` : Dispatch tag indicating the locally isothermal EOS.
- `u::T` : Specific internal energy.
- `r::T` : Radial position.
- `cs0::T` : Reference sound speed at `r = 1`.
- `q::T` : Power-law exponent controlling radial dependence.

# Returns
- `T` : The computed specific enthalpy at radius `r`, or `NaN` if input is unphysical.
"""
@inline function Enthalpy(::Type{LocallyIsothermal}, u::T, r::T, cs0::T, q::T) :: T where {T<:AbstractFloat}
    if u < zero(T) || r <= zero(T) || cs0 < zero(T)
        return T(NaN)
    end
    return u + (cs0 * r^(-q))^2
end

"""
    Enthalpy(::Type{LocallyIsothermal}, u::AbstractFloat, r::AbstractFloat, cs0::AbstractFloat, q::AbstractFloat)

Compute the specific enthalpy for a locally isothermal gas from the specific
internal energy `u`, radial position `r`, reference sound speed `cs0`, and
radial power-law index `q`, after promoting all inputs to a common floating-point
type.

The enthalpy is evaluated as

    h(r) = u + [ cs0 r^(-q) ]^2.

Returns `NaN` if `u < 0`, `r ≤ 0`, or `cs0 < 0` after promotion.

# Parameters
- `::Type{LocallyIsothermal}` : Dispatch tag indicating the locally isothermal EOS.
- `u::AbstractFloat` : Specific internal energy.
- `r::AbstractFloat` : Radial position.
- `cs0::AbstractFloat` : Reference sound speed at `r = 1`.
- `q::AbstractFloat` : Power-law exponent controlling radial dependence.

# Returns
- `promote_type(typeof(u), typeof(r), typeof(cs0), typeof(q))` : The computed
  specific enthalpy in the promoted floating-point type, or `NaN` if input is unphysical.
"""
@inline function Enthalpy(::Type{LocallyIsothermal}, u::AbstractFloat, r::AbstractFloat, cs0::AbstractFloat, q::AbstractFloat)
    up, rp, cs0p, qp = promote(u, r, cs0, q)
    return Enthalpy(LocallyIsothermal, up, rp, cs0p, qp)
end
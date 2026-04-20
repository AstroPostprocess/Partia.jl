"""
    SoundSpeed(::Type{LocallyIsothermal}, r::T, cs0::T, q::T) :: T where {T<:AbstractFloat}

Compute the locally isothermal sound speed profile, defined as

    c_s(r) = cs0 * r^(-q),

where `cs0` is a reference sound speed and `q` is the radial power-law index.
Returns `NaN` if `r ≤ 0` or `cs0 < 0`.

# Parameters
- `::Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal sound speed.
- `r::T` : Radial position.
- `cs0::T` : Reference sound speed at `r = 1`.
- `q::T` : Power-law exponent controlling radial dependence.

# Returns
- `T` : The locally isothermal sound speed at radius `r`, or `NaN` if input is unphysical.
"""
@inline function SoundSpeed(::Type{LocallyIsothermal}, r::T, cs0::T, q::T) :: T where {T<:AbstractFloat}
    if r <= zero(T) || cs0 < zero(T)
        return T(NaN)
    end
    return cs0 * r^(-q)
end

"""
    SoundSpeed(::Type{LocallyIsothermal}, r::AbstractFloat, cs0::AbstractFloat, q::AbstractFloat)

Compute the locally isothermal sound speed with automatic type promotion, defined as

    c_s(r) = cs0 * r^(-q),

where `cs0` is a reference sound speed and `q` is the radial power-law index.
Returns `NaN` if `r ≤ 0` or `cs0 < 0`.

# Parameters
- `::Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal sound speed.
- `r::AbstractFloat` : Radial position.
- `cs0::AbstractFloat` : Reference sound speed at `r = 1`.
- `q::AbstractFloat` : Power-law exponent controlling radial dependence.

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the
  locally isothermal sound speed at radius `r`, or `NaN` if input is unphysical.
"""
@inline function SoundSpeed(::Type{LocallyIsothermal}, r::AbstractFloat, cs0::AbstractFloat, q::AbstractFloat)
    rp, cs0p, qp = promote(r, cs0, q)
    T = typeof(rp)
    if rp <= zero(T) || cs0p < zero(T)
        return T(NaN)
    end
    return cs0p * rp^(-qp)
end

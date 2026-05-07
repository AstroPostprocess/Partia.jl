"""
    Temperature( :: Type{LocallyIsothermal}, :: Type{SIUnit}, r :: T, cs0 :: T, q :: T, μ :: T) :: T where {T <: AbstractFloat}

Compute the locally isothermal gas temperature in SI units from the radial position,
reference sound speed, radial power-law index, and mean molecular weight. The formula is

    T(r) = (m_p / k_B) * μ [ cs0 * r^(-q) ]²,

where `cs0` is the reference sound speed at `r = 1` and `q` is the radial power-law index.
The constant `(m_p / k_B)` is precomputed as `0.00012114751277768644 [K kg J⁻¹]`.
Returns `NaN` if `μ < 0`, `r ≤ 0`, or `cs0 < 0`.

# Parameters
- ` :: Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal temperature calculation.
- ` :: Type{SIUnit}` : Dispatch tag specifying SI unit convention.
- `r :: T` : Radial position.
- `cs0 :: T` : Reference sound speed at `r = 1`.
- `q :: T` : Power-law exponent controlling radial dependence.
- `μ :: T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature( :: Type{LocallyIsothermal}, :: Type{SIUnit}, r :: T, cs0 :: T, q :: T, μ :: T) :: T where {T <: AbstractFloat}
    mplkB = T(0.00012114751277768644)
    if μ < zero(T) || r <= zero(T) || cs0 < zero(T)
        return T(NaN)
    end
    cs = cs0 * r^(-q)
    return mplkB * μ * cs^2
end

"""
    Temperature( :: Type{LocallyIsothermal}, :: Type{CGSUnit}, r :: T, cs0 :: T, q :: T, μ :: T) :: T where {T <: AbstractFloat}

Compute the locally isothermal gas temperature in CGS units from the radial position,
reference sound speed, radial power-law index, and mean molecular weight. The formula is

    T(r) = (m_p / k_B) * μ [ cs0 * r^(-q) ]²,

where `cs0` is the reference sound speed at `r = 1` and `q` is the radial power-law index.
The constant `(m_p / k_B)` is precomputed as `1.2114751277768644e-8 [K g erg⁻¹]`.
Returns `NaN` if `μ < 0`, `r ≤ 0`, or `cs0 < 0`.

# Parameters
- ` :: Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal temperature calculation.
- ` :: Type{CGSUnit}` : Dispatch tag specifying CGS unit convention.
- `r :: T` : Radial position.
- `cs0 :: T` : Reference sound speed at `r = 1` (cm s⁻¹).
- `q :: T` : Power-law exponent controlling radial dependence.
- `μ :: T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature( :: Type{LocallyIsothermal}, :: Type{CGSUnit}, r :: T, cs0 :: T, q :: T, μ :: T) :: T where {T <: AbstractFloat}
    mplkB = T(1.2114751277768644e-8)
    if μ < zero(T) || r <= zero(T) || cs0 < zero(T)
        return T(NaN)
    end
    cs = cs0 * r^(-q)
    return mplkB * μ * cs^2
end

"""
    Temperature( :: Type{LocallyIsothermal}, :: Type{GalacticUnit}, r :: T, cs0 :: T, q :: T, μ :: T) :: T where {T <: AbstractFloat}

Compute the locally isothermal gas temperature in Galactic units from the radial position,
reference sound speed, radial power-law index, and mean molecular weight. The formula is

    T(r) = (m_p / k_B) * μ [ cs0 * r^(-q) ]²,

where `cs0` is the reference sound speed at `r = 1` and `q` is the radial power-law index.
The constant `(m_p / k_B)` is precomputed as `121.14751277768644 [K s² km⁻²]`.
Returns `NaN` if `μ < 0`, `r ≤ 0`, or `cs0 < 0`.

# Parameters
- ` :: Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal temperature calculation.
- ` :: Type{GalacticUnit}` : Dispatch tag specifying Galactic unit convention.
- `r :: T` : Radial position.
- `cs0 :: T` : Reference sound speed at `r = 1` (km s⁻¹).
- `q :: T` : Power-law exponent controlling radial dependence.
- `μ :: T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature( :: Type{LocallyIsothermal}, :: Type{GalacticUnit}, r :: T, cs0 :: T, q :: T, μ :: T) :: T where {T <: AbstractFloat}
    mplkB = T(121.14751277768644)
    if μ < zero(T) || r <= zero(T) || cs0 < zero(T)
        return T(NaN)
    end
    cs = cs0 * r^(-q)
    return mplkB * μ * cs^2
end

"""
    Temperature( :: Type{LocallyIsothermal}, :: Type{SIUnit}, r :: AbstractFloat, cs0 :: AbstractFloat, q :: AbstractFloat, μ :: AbstractFloat)

Compute the locally isothermal gas temperature in SI units with automatic type promotion, using

    T(r) = (m_p / k_B) * μ [ cs0 * r^(-q) ]²,

where `cs0` is the reference sound speed at `r = 1`, `q` is the radial power-law index,
and `r` is the radial position.
The constant `(m_p / k_B)` is precomputed as `0.00012114751277768644 [K kg J⁻¹]`.
Returns `NaN` if `μ < 0`, `r ≤ 0`, or `cs0 < 0`.

# Parameters
- ` :: Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal temperature calculation.
- ` :: Type{SIUnit}` : Dispatch tag specifying SI unit convention.
- `r :: AbstractFloat` : Radial position.
- `cs0 :: AbstractFloat` : Reference sound speed at `r = 1` (m s⁻¹).
- `q :: AbstractFloat` : Power-law exponent controlling radial dependence.
- `μ :: AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature( :: Type{LocallyIsothermal}, :: Type{SIUnit}, r :: AbstractFloat, cs0 :: AbstractFloat, q :: AbstractFloat, μ :: AbstractFloat)
    rp, cs0p, qp, μp = promote(r, cs0, q, μ)
    T = typeof(rp)
    mplkB = T(0.00012114751277768644)
    if μp < zero(T) || rp <= zero(T) || cs0p < zero(T)
        return T(NaN)
    end
    cs = cs0p * rp^(-qp)
    return mplkB * μp * cs^2
end

"""
    Temperature( :: Type{LocallyIsothermal}, :: Type{CGSUnit}, r :: AbstractFloat, cs0 :: AbstractFloat, q :: AbstractFloat, μ :: AbstractFloat)

Compute the locally isothermal gas temperature in CGS units with automatic type promotion, using

    T(r) = (m_p / k_B) * μ [ cs0 * r^(-q) ]²,

where `cs0` is the reference sound speed at `r = 1`, `q` is the radial power-law index,
and `r` is the radial position.
The constant `(m_p / k_B)` is precomputed as `1.2114751277768644e-8 [K g erg⁻¹]`.
Returns `NaN` if `μ < 0`, `r ≤ 0`, or `cs0 < 0`.

# Parameters
- ` :: Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal temperature calculation.
- ` :: Type{CGSUnit}` : Dispatch tag specifying CGS unit convention.
- `r :: AbstractFloat` : Radial position.
- `cs0 :: AbstractFloat` : Reference sound speed at `r = 1` (cm s⁻¹).
- `q :: AbstractFloat` : Power-law exponent controlling radial dependence.
- `μ :: AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature( :: Type{LocallyIsothermal}, :: Type{CGSUnit}, r :: AbstractFloat, cs0 :: AbstractFloat, q :: AbstractFloat, μ :: AbstractFloat)
    rp, cs0p, qp, μp = promote(r, cs0, q, μ)
    T = typeof(rp)
    mplkB = T(1.2114751277768644e-8)
    if μp < zero(T) || rp <= zero(T) || cs0p < zero(T)
        return T(NaN)
    end
    cs = cs0p * rp^(-qp)
    return mplkB * μp * cs^2
end

"""
    Temperature( :: Type{LocallyIsothermal}, :: Type{GalacticUnit}, r :: AbstractFloat, cs0 :: AbstractFloat, q :: AbstractFloat, μ :: AbstractFloat)

Compute the locally isothermal gas temperature in Galactic units with automatic type promotion, using

    T(r) = (m_p / k_B) * μ [ cs0 * r^(-q) ]²,

where `cs0` is the reference sound speed at `r = 1`, `q` is the radial power-law index,
and `r` is the radial position.
The constant `(m_p / k_B)` is precomputed as `121.14751277768644 [K s² km⁻²]`.
Returns `NaN` if `μ < 0`, `r ≤ 0`, or `cs0 < 0`.

# Parameters
- ` :: Type{LocallyIsothermal}` : Dispatch tag indicating locally isothermal temperature calculation.
- ` :: Type{GalacticUnit}` : Dispatch tag specifying Galactic unit convention.
- `r :: AbstractFloat` : Radial position.
- `cs0 :: AbstractFloat` : Reference sound speed at `r = 1` (km s⁻¹).
- `q :: AbstractFloat` : Power-law exponent controlling radial dependence.
- `μ :: AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature( :: Type{LocallyIsothermal}, :: Type{GalacticUnit}, r :: AbstractFloat, cs0 :: AbstractFloat, q :: AbstractFloat, μ :: AbstractFloat)
    rp, cs0p, qp, μp = promote(r, cs0, q, μ)
    T = typeof(rp)
    mplkB = T(121.14751277768644)
    if μp < zero(T) || rp <= zero(T) || cs0p < zero(T)
        return T(NaN)
    end
    cs = cs0p * rp^(-qp)
    return mplkB * μp * cs^2
end

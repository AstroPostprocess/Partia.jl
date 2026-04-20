"""
    Temperature(::Type{Isothermal}, ::Type{SIUnit}, cs::T, μ::T) :: T where {T<:AbstractFloat}

Compute the isothermal gas temperature in SI units from the isothermal sound speed and
mean molecular weight. The formula is

    T = (m_p / k_B) * μ c_s²,

where `c_s` is the isothermal sound speed.
The constant `(m_p / k_B)` is precomputed as `0.00012114751277768644 [K kg J⁻¹]`.
Returns `NaN` if `μ < 0` or `c_s < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal temperature calculation.
- `::Type{SIUnit}` : Dispatch tag specifying SI unit convention.
- `cs::T` : Isothermal sound speed.
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Isothermal}, ::Type{SIUnit}, cs::T, μ::T) :: T where {T<:AbstractFloat}
    mplkB = T(0.00012114751277768644)
    if μ < zero(T) || cs < zero(T)
        return T(NaN)
    end
    return mplkB * μ * cs^2
end

"""
    Temperature(::Type{Isothermal}, ::Type{CGSUnit}, cs::T, μ::T) :: T where {T<:AbstractFloat}

Compute the isothermal gas temperature in CGS units from the isothermal sound speed
and mean molecular weight. The formula is

    T = (m_p / k_B) * μ c_s²,

where `c_s` is the isothermal sound speed in cm s⁻¹.
The constant `(m_p / k_B)` is precomputed as `1.2114751277768644e-8 [K g erg⁻¹]`.
Returns `NaN` if `μ < 0` or `c_s < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal temperature calculation.
- `::Type{CGSUnit}` : Dispatch tag specifying CGS unit convention.
- `cs::T` : Isothermal sound speed (cm s⁻¹).
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Isothermal}, ::Type{CGSUnit}, cs::T, μ::T) :: T where {T<:AbstractFloat}
    mplkB = T(1.2114751277768644e-8)
    if μ < zero(T) || cs < zero(T)
        return T(NaN)
    end
    return mplkB * μ * cs^2
end

"""
    Temperature(::Type{Isothermal}, ::Type{GalacticUnit}, cs::T, μ::T) :: T where {T<:AbstractFloat}

Compute the isothermal gas temperature in Galactic units from the isothermal sound speed
and mean molecular weight. The formula is

    T = (m_p / k_B) * μ c_s²,

where `c_s` is the isothermal sound speed in km s⁻¹.
The constant `(m_p / k_B)` is precomputed as `121.14751277768644 [K s² km⁻²]`.
Returns `NaN` if `μ < 0` or `c_s < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal temperature calculation.
- `::Type{GalacticUnit}` : Dispatch tag specifying Galactic unit convention.
- `cs::T` : Isothermal sound speed (km s⁻¹).
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Isothermal}, ::Type{GalacticUnit}, cs::T, μ::T) :: T where {T<:AbstractFloat}
    mplkB = T(121.14751277768644)
    if μ < zero(T) || cs < zero(T)
        return T(NaN)
    end
    return mplkB * μ * cs^2
end

"""
    Temperature(::Type{Isothermal}, ::Type{SIUnit}, cs::AbstractFloat, μ::AbstractFloat)

Compute the isothermal gas temperature in SI units with automatic type promotion, using

    T = (m_p / k_B) * μ c_s²,

where `c_s` is the isothermal sound speed in m s⁻¹.
The constant `(m_p / k_B)` is precomputed as `0.00012114751277768644 [K kg J⁻¹]`.
Returns `NaN` if `μ < 0` or `c_s < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal temperature calculation.
- `::Type{SIUnit}` : Dispatch tag specifying SI unit convention.
- `cs::AbstractFloat` : Isothermal sound speed (m s⁻¹).
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Isothermal}, ::Type{SIUnit}, cs::AbstractFloat, μ::AbstractFloat)
    csp, μp = promote(cs, μ)
    T = typeof(csp)
    mplkB = T(0.00012114751277768644)
    if μp < zero(T) || csp < zero(T)
        return T(NaN)
    end
    return mplkB * μp * csp^2
end

"""
    Temperature(::Type{Isothermal}, ::Type{CGSUnit}, cs::AbstractFloat, μ::AbstractFloat)

Compute the isothermal gas temperature in CGS units with automatic type promotion, using

    T = (m_p / k_B) * μ c_s²,

where `c_s` is the isothermal sound speed in cm s⁻¹.
The constant `(m_p / k_B)` is precomputed as `1.2114751277768644e-8 [K g erg⁻¹]`.
Returns `NaN` if `μ < 0` or `c_s < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal temperature calculation.
- `::Type{CGSUnit}` : Dispatch tag specifying CGS unit convention.
- `cs::AbstractFloat` : Isothermal sound speed (cm s⁻¹).
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Isothermal}, ::Type{CGSUnit}, cs::AbstractFloat, μ::AbstractFloat)
    csp, μp = promote(cs, μ)
    T = typeof(csp)
    mplkB = T(1.2114751277768644e-8)
    if μp < zero(T) || csp < zero(T)
        return T(NaN)
    end
    return mplkB * μp * csp^2
end

"""
    Temperature(::Type{Isothermal}, ::Type{GalacticUnit}, cs::AbstractFloat, μ::AbstractFloat)

Compute the isothermal gas temperature in Galactic units with automatic type promotion, using

    T = (m_p / k_B) * μ c_s²,

where `c_s` is the isothermal sound speed in km s⁻¹.
The constant `(m_p / k_B)` is precomputed as `121.14751277768644 [K s² km⁻²]`.
Returns `NaN` if `μ < 0` or `c_s < 0`.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal temperature calculation.
- `::Type{GalacticUnit}` : Dispatch tag specifying Galactic unit convention.
- `cs::AbstractFloat` : Isothermal sound speed (km s⁻¹).
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Isothermal}, ::Type{GalacticUnit}, cs::AbstractFloat, μ::AbstractFloat)
    csp, μp = promote(cs, μ)
    T = typeof(csp)
    mplkB = T(121.14751277768644)
    if μp < zero(T) || csp < zero(T)
        return T(NaN)
    end
    return mplkB * μp * csp^2
end

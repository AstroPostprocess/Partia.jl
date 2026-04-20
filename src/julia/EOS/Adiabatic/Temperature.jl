"""
    Temperature(::Type{Adiabatic}, ::Type{SIUnit}, u::T, γ::T, μ::T) :: T where {T<:AbstractFloat}

Compute the adiabatic gas temperature in SI units from the specific internal energy,
adiabatic index, and mean molecular weight. The formula is

    T = (m_p / k_B) * μ (γ - 1) u,

where `m_p` is the proton mass and `k_B` is the Boltzmann constant.
Here, the constant factor `(m_p / k_B)` is precomputed as `0.00012114751277768644 [K kg J⁻¹]` in SI units.
Returns `NaN` if `μ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic temperature calculation.
- `::Type{SIUnit}` : Dispatch tag specifying SI unit convention.
- `u::T` : Specific internal energy (m² s⁻²).
- `γ::T` : Adiabatic index.
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Adiabatic}, ::Type{SIUnit}, u::T, γ::T, μ::T) :: T where {T<:AbstractFloat}
    mplkB = T(0.00012114751277768644)
    if μ < zero(T) || u < zero(T) || γ < one(T)
        return T(NaN)
    end
    return mplkB * μ * (γ - one(T)) * u
end

"""
    Temperature(::Type{Adiabatic}, ::Type{CGSUnit}, u::T, γ::T, μ::T) :: T where {T<:AbstractFloat}

Compute the adiabatic gas temperature in CGS units from the specific internal energy,
adiabatic index, and mean molecular weight. The formula is

    T = (m_p / k_B) * μ (γ - 1) u,

where `u` is the specific internal energy in cm² s⁻².
The constant `(m_p / k_B)` is precomputed as `1.2114751277768644e-8 [K g erg⁻¹]`.
Returns `NaN` if `μ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic temperature calculation.
- `::Type{CGSUnit}` : Dispatch tag specifying CGS unit convention.
- `u::T` : Specific internal energy (cm² s⁻²).
- `γ::T` : Adiabatic index.
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Adiabatic}, ::Type{CGSUnit}, u::T, γ::T, μ::T) :: T where {T<:AbstractFloat}
    mplkB = T(1.2114751277768644e-8)
    if μ < zero(T) || u < zero(T) || γ < one(T)
        return T(NaN)
    end
    return mplkB * μ * (γ - one(T)) * u
end

"""
    Temperature(::Type{Adiabatic}, ::Type{GalacticUnit}, u::T, γ::T, μ::T) :: T where {T<:AbstractFloat}

Compute the adiabatic gas temperature in Galactic units from the specific internal energy,
adiabatic index, and mean molecular weight. The formula is

    T = (m_p / k_B) * μ (γ - 1) u,

where `u` is the specific internal energy in km² s⁻².
The constant `(m_p / k_B)` is precomputed as `121.14751277768644 [K s² km⁻²]`.
Returns `NaN` if `μ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic temperature calculation.
- `::Type{GalacticUnit}` : Dispatch tag specifying Galactic unit convention.
- `u::T` : Specific internal energy (km² s⁻²).
- `γ::T` : Adiabatic index.
- `μ::T` : Mean molecular weight (dimensionless).

# Returns
- `T` : Temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Adiabatic}, ::Type{GalacticUnit}, u::T, γ::T, μ::T) :: T where {T<:AbstractFloat}
    mplkB = T(121.14751277768644)
    if μ < zero(T) || u < zero(T) || γ < one(T)
        return T(NaN)
    end
    return mplkB * μ * (γ - one(T)) * u
end

"""
    Temperature(::Type{Adiabatic}, ::Type{SIUnit}, u::AbstractFloat, γ::AbstractFloat, μ::AbstractFloat)

Compute the adiabatic gas temperature in SI units with automatic type promotion, using

    T = (m_p / k_B) * μ (γ - 1) u,

where `u` is the specific internal energy in m² s⁻².
The constant `(m_p / k_B)` is precomputed as `0.00012114751277768644 [K kg J⁻¹]`.
Returns `NaN` if `μ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic temperature calculation.
- `::Type{SIUnit}` : Dispatch tag specifying SI unit convention.
- `u::AbstractFloat` : Specific internal energy (m² s⁻²).
- `γ::AbstractFloat` : Adiabatic index.
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Adiabatic}, ::Type{SIUnit}, u::AbstractFloat, γ::AbstractFloat, μ::AbstractFloat)
    up, γp, μp = promote(u, γ, μ)
    T = typeof(up)
    mplkB = T(0.00012114751277768644)
    if μp < zero(T) || up < zero(T) || γp < one(T)
        return T(NaN)
    end
    return mplkB * μp * (γp - one(T)) * up
end

"""
    Temperature(::Type{Adiabatic}, ::Type{CGSUnit}, u::AbstractFloat, γ::AbstractFloat, μ::AbstractFloat)

Compute the adiabatic gas temperature in CGS units with automatic type promotion, using

    T = (m_p / k_B) * μ (γ - 1) u,

where `u` is the specific internal energy in cm² s⁻².
The constant `(m_p / k_B)` is precomputed as `1.2114751277768644e-8 [K g erg⁻¹]`.
Returns `NaN` if `μ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic temperature calculation.
- `::Type{CGSUnit}` : Dispatch tag specifying CGS unit convention.
- `u::AbstractFloat` : Specific internal energy (cm² s⁻²).
- `γ::AbstractFloat` : Adiabatic index.
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Adiabatic}, ::Type{CGSUnit}, u::AbstractFloat, γ::AbstractFloat, μ::AbstractFloat)
    up, γp, μp = promote(u, γ, μ)
    T = typeof(up)
    mplkB = T(1.2114751277768644e-8)
    if μp < zero(T) || up < zero(T) || γp < one(T)
        return T(NaN)
    end
    return mplkB * μp * (γp - one(T)) * up
end

"""
    Temperature(::Type{Adiabatic}, ::Type{GalacticUnit}, u::AbstractFloat, γ::AbstractFloat, μ::AbstractFloat)

Compute the adiabatic gas temperature in Galactic units with automatic type promotion, using

    T = (m_p / k_B) * μ (γ - 1) u,

where `u` is the specific internal energy in km² s⁻².
The constant `(m_p / k_B)` is precomputed as `121.14751277768644 [K s² km⁻²]`.
Returns `NaN` if `μ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic temperature calculation.
- `::Type{GalacticUnit}` : Dispatch tag specifying Galactic unit convention.
- `u::AbstractFloat` : Specific internal energy (km² s⁻²).
- `γ::AbstractFloat` : Adiabatic index.
- `μ::AbstractFloat` : Mean molecular weight (dimensionless).

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the
  temperature in Kelvin, or `NaN` if input is unphysical.
"""
@inline function Temperature(::Type{Adiabatic}, ::Type{GalacticUnit}, u::AbstractFloat, γ::AbstractFloat, μ::AbstractFloat)
    up, γp, μp = promote(u, γ, μ)
    T = typeof(up)
    mplkB = T(121.14751277768644)
    if μp < zero(T) || up < zero(T) || γp < one(T)
        return T(NaN)
    end
    return mplkB * μp * (γp - one(T)) * up
end

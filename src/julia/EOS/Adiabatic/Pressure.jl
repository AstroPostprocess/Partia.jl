"""
    Pressure(::Type{Adiabatic}, ρ::T, u::T, γ::T) :: T where {T<:AbstractFloat}

Compute the adiabatic gas pressure using the equation of state

    P = (γ - 1) ρ u,

where `ρ` is the mass density, `u` is the specific internal energy, and `γ` is the adiabatic index.
Returns `NaN` if `ρ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic pressure calculation.
- `ρ::T` : Mass density.
- `u::T` : Specific internal energy.
- `γ::T` : Adiabatic index.

# Returns
- `T` : The computed pressure, or `NaN` if input is unphysical.
"""
@inline function Pressure(::Type{Adiabatic}, ρ::T, u::T, γ::T) :: T where {T<:AbstractFloat}
    if ρ < zero(T) || u < zero(T) || γ < one(T)
        return T(NaN)
    end
    return (γ - one(T)) * ρ * u
end

"""
    Pressure(::Type{Adiabatic}, ρ::AbstractFloat, u::AbstractFloat, γ::AbstractFloat)

Compute the adiabatic gas pressure with automatic type promotion, using the equation of state

    P = (γ - 1) ρ u,

where `ρ` is the mass density, `u` is the specific internal energy, and `γ` is the adiabatic index.
Returns `NaN` if `ρ < 0`, `u < 0`, or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic pressure calculation.
- `ρ::AbstractFloat` : Mass density.
- `u::AbstractFloat` : Specific internal energy.
- `γ::AbstractFloat` : Adiabatic index.

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the
  computed pressure, or `NaN` if input is unphysical.
"""
@inline function Pressure(::Type{Adiabatic}, ρ::AbstractFloat, u::AbstractFloat, γ::AbstractFloat)
    ρp, up, γp = promote(ρ, u, γ)
    T = typeof(ρp)
    if ρp < zero(T) || up < zero(T) || γp < one(T)
        return T(NaN)
    end
    return (γp - one(T)) * ρp * up
end

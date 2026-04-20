"""
    SoundSpeed(::Type{Adiabatic}, u::T, γ::T) :: T where {T<:AbstractFloat}

Compute the adiabatic sound speed from the specific internal energy `u` and
adiabatic index `γ`, using

    c_s = √[ γ (γ - 1) u ].

Returns `NaN` if `u < 0` or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic sound speed.
- `u::T` : Specific internal energy.
- `γ::T` : Adiabatic index.

# Returns
- `T` : The computed sound speed, or `NaN` if input is unphysical.
"""
@inline function SoundSpeed(::Type{Adiabatic}, u::T, γ::T) :: T where {T<:AbstractFloat}
    if u < zero(T) || γ < one(T)
        return T(NaN)
    end
    return sqrt(γ * (γ - one(T)) * u)
end

"""
    SoundSpeed(::Type{Adiabatic}, u::AbstractFloat, γ::AbstractFloat)

Compute the adiabatic sound speed with automatic type promotion, using

    c_s = √[ γ (γ - 1) u ],

where `u` is the specific internal energy and `γ` is the adiabatic index.
Returns `NaN` if `u < 0` or `γ < 1`.

# Parameters
- `::Type{Adiabatic}` : Dispatch tag indicating adiabatic sound speed.
- `u::AbstractFloat` : Specific internal energy.
- `γ::AbstractFloat` : Adiabatic index.

# Returns
- `AbstractFloat` : The promoted floating-point type of the inputs, representing the
  computed sound speed, or `NaN` if input is unphysical.
"""
@inline function SoundSpeed(::Type{Adiabatic}, u::AbstractFloat, γ::AbstractFloat)
    up, γp = promote(u, γ)
    T = typeof(up)
    if up < zero(T) || γp < one(T)
        return T(NaN)
    end
    return sqrt(γp * (γp - one(T)) * up)
end

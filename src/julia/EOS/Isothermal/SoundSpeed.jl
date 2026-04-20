"""
    SoundSpeed(::Type{Isothermal}, cs::T)  where {T<:AbstractFloat}

Return the isothermal sound speed. In an isothermal equation of state, the
sound speed is constant and equal to the input value.

# Parameters
- `::Type{Isothermal}` : Dispatch tag indicating isothermal sound speed.
- `cs::T` : Prescribed constant sound speed.

# Returns
- `T` : The same value `cs`, representing the isothermal sound speed.
"""
@inline SoundSpeed(::Type{Isothermal}, cs::T) where {T<:AbstractFloat} = cs

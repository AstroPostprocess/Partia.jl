######################################################################################

# Unit Type

######################################################################################
"""Abstract supertype for unit-system dispatch tags used by EOS functions."""
abstract type AbstractUnit end

"""Dispatch tag selecting SI units."""
struct SIUnit <: AbstractUnit end

"""Dispatch tag selecting centimetre-gram-second (CGS) units."""
struct CGSUnit <: AbstractUnit end

## Astronomical Type
"""Abstract supertype for astronomical unit-system dispatch tags."""
abstract type AstronomicalUnit <: AbstractUnit end

"""Stellar units with mass in solar masses and distance in solar radii."""
struct StarUnit <: AstronomicalUnit end             # mass: M⊙, distance: R⊙

"""Solar-system units with mass in solar masses and distance in astronomical units."""
struct SolarSystemUnit <: AstronomicalUnit end      # mass: M⊙, distance: AU

"""Galactic units with velocity in km/s and distance in kpc."""
struct GalacticUnit <: AstronomicalUnit end         # velocity: km/s, distance: kpc (time ~ 0.978 Gyr)

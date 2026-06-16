######################################################################################

# Unit Type

######################################################################################
abstract type AbstractUnit end
struct SIUnit <: AbstractUnit end
struct CGSUnit <: AbstractUnit end
## Astronomical Type
abstract type AstronomicalUnit <: AbstractUnit end
struct StarUnit <: AstronomicalUnit end             # mass: M⊙, distance: R⊙
struct SolarSystemUnit <: AstronomicalUnit end      # mass: M⊙, distance: AU
struct GalacticUnit <: AstronomicalUnit end         # velocity: km/s, distance: kpc (time ~ 0.978 Gyr)
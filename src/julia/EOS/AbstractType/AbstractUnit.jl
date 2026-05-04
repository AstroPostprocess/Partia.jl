######################################################################################

# Unit Type

######################################################################################
abstract type AbstractUnit end
struct SIUnit <: AbstractUnit end
struct CGSUnit <: AbstractUnit end
## Astonomical Type
abstract type AstronomicalUnit <: AbstractUnit end
struct StarUnit <: AstronomicalUnit end             # mass: M⊙, dist: AU
struct GalacticUnit <: AstronomicalUnit end         # velocity: km/s, distance: kpc
"""Abstract supertype for line-sampling beam-model dispatch tags."""
abstract type AbstractBeamModel end

"""Beam model whose sample lines all share one direction."""
struct ParallelBeam <: AbstractBeamModel end

"""Beam model whose sample lines originate from a common pinhole."""
struct Pinhole     <: AbstractBeamModel end

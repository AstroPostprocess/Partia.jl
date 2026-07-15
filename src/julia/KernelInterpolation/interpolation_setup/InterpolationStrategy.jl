"""Abstract supertype for interpolation-strategy dispatch tags."""
abstract type AbstractInterpolationStrategy end

"""Gather interpolation using a sample-side smoothing length."""
struct itpGather <: AbstractInterpolationStrategy end

"""Scatter interpolation using particle-side smoothing lengths."""
struct itpScatter <: AbstractInterpolationStrategy end



"""
    AbstractSamples{D, TF} <: AbstractGrid{TF}

Abstract supertype for unstructured sample containers embedded in `D`
dimensions with floating-point value type `TF`.
"""
abstract type AbstractSamples{D, TF <: AbstractFloat} <: AbstractGrid{TF} end

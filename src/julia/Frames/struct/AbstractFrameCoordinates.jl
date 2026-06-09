

"""
    AbstractFrameCoordinates

Abstract supertype for frame-coordinate dispatch tags.
"""
abstract type AbstractFrameCoordinates end

"""
    LocalCoordinates

Dispatch tag for interpreting frame displacements along the frame's current local directions.
"""
struct LocalCoordinates  <: AbstractFrameCoordinates end

"""
    GlobalCoordinates

Dispatch tag for interpreting frame displacements along the fixed global axes.
"""
struct GlobalCoordinates <: AbstractFrameCoordinates end

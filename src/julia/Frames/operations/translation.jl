"""
    translate!( :: Type{GlobalCoordinates}, frame :: Frame{TF}, Δx :: TF, Δy :: TF, Δz :: TF) where {TF <: AbstractFloat}

Translate `frame` by a displacement expressed along the fixed global axes.
`GlobalCoordinates` interprets the displacement components along the fixed global x, y, and z axes.
This method modifies only `frame.x`; it does not modify `frame.Q`.

# Parameters
- `GlobalCoordinates`: Dispatch tag selecting global-coordinate translation.
- `frame`: Frame whose position is updated.
- `Δx`: Displacement component along the fixed global x axis.
- `Δy`: Displacement component along the fixed global y axis.
- `Δz`: Displacement component along the fixed global z axis.

# Returns
- `nothing`: `frame` is updated in place.
"""
@inline function translate!( :: Type{GlobalCoordinates}, frame :: Frame{TF}, Δx :: TF, Δy :: TF, Δz :: TF) where {TF <: AbstractFloat}
    frame.x[1] += Δx
    frame.x[2] += Δy
    frame.x[3] += Δz
    return nothing
end

"""
    translate!( :: Type{LocalCoordinates}, frame :: Frame{TF}, Δr :: TF, Δf :: TF, Δu :: TF) where {TF <: AbstractFloat}

Translate `frame` by a displacement expressed along the frame's current local directions.
`LocalCoordinates` interprets the displacement components along the frame's current right, up, and forward directions.
The current local directions are obtained by rotating `frame.r0`, `frame.f0` and `frame.u0` using `frame.Q`.
This method modifies only `frame.x`; it does not modify `frame.Q`.

# Parameters
- `LocalCoordinates`: Dispatch tag selecting local-coordinate translation.
- `frame`: Frame whose position is updated.
- `Δr`: Displacement component along the frame's current right direction.
- `Δf`: Displacement component along the frame's current forward direction.
- `Δu`: Displacement component along the frame's current up direction.

# Returns
- `nothing`: `frame` is updated in place.
"""
@inline function translate!( :: Type{LocalCoordinates}, frame :: Frame{TF}, Δr :: TF, Δf :: TF, Δu :: TF,) where {TF <: AbstractFloat}
    Q = frame.Q

    # Get the current local basis vectors in global coordinates
    r = _rotate(Q, frame.r0)
    u = _rotate(Q, frame.u0)
    f = _rotate(Q, frame.f0)

    # Translate the frame position along the current local basis vectors
    frame.x[1] += Δr * r[1] + Δu * u[1] + Δf * f[1]
    frame.x[2] += Δr * r[2] + Δu * u[2] + Δf * f[2]
    frame.x[3] += Δr * r[3] + Δu * u[3] + Δf * f[3]

    return nothing
end

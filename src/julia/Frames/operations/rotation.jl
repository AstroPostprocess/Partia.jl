"""
    rotate!(frame::Frame{TF}, yaw::TF, pitch::TF, roll::TF) where {TF <: AbstractFloat}

Rotate `frame` by updating its orientation quaternion.
The angles are applied as yaw about `frame.u0`, pitch about `frame.r0`, and roll about `frame.f0`, in that order.
The updated quaternion is normalized after the rotation is accumulated.
This method modifies only `frame.Q`; it does not modify `frame.x`.

# Parameters
- `frame`: Frame whose orientation is updated.
- `yaw`: Rotation angle applied about the frame's initial up direction.
- `pitch`: Rotation angle applied about the frame's initial right direction.
- `roll`: Rotation angle applied about the frame's initial forward direction.
"""
@inline function rotate!(frame :: Frame{TF}, yaw :: TF, pitch :: TF, roll :: TF) where {TF <: AbstractFloat}
    # Get the operator(quaternion) of rotation
    Qyaw   = _rotation(frame.u0, yaw)
    Qpitch = _rotation(frame.r0, pitch)
    Qroll  = _rotation(frame.f0, roll)

    # Rotate Q in frame
    frame.Q = frame.Q * Qyaw * Qpitch * Qroll
    frame.Q /= norm(frame.Q)

    return nothing
end

"""
    rotate_forward_to!(frame::Frame{TF}, target_f::NTuple{3, TF}) where {TF <: AbstractFloat}
    rotate_forward_to!(frame::Frame{TF}, target_f::AbstractVector{TF}) where {TF <: AbstractFloat}

Rotate `frame` so that its current forward direction is aligned with `target_f`.
The operation applies the minimal global rotation that maps `frame_forward(frame)`
onto the normalized target direction. It modifies only `frame.Q`; it does not
modify `frame.x`.

If the current forward direction and `target_f` are opposite, the frame is
rotated by π about its current up direction to choose a stable 180-degree
alignment.

# Parameters
- `frame`: Frame whose orientation is updated.
- `target_f`: Target forward direction in global coordinates.
"""
@inline function rotate_forward_to!(frame :: Frame{TF}, target_f :: NTuple{3,TF}) where {TF <: AbstractFloat}
    return rotate_forward_to!(frame, SVector{3,TF}(target_f))
end

@inline function rotate_forward_to!(frame :: Frame{TF}, target_f :: AbstractVector{TF}) where {TF <: AbstractFloat}
    length(target_f) == 3 || throw(DimensionMismatch("target_f must have length 3"))

    target = SVector{3,TF}(target_f)
    ntarget = norm(target)
    iszero(ntarget) && throw(ArgumentError("target_f must be nonzero."))
    target /= ntarget

    current = _rotate(frame.Q, frame.f0)
    d = clamp(dot(current, target), -one(TF), one(TF))
    tol = TF(16) * eps(TF)

    if d >= one(TF) - tol
        return nothing
    elseif d <= -one(TF) + tol
        axis = _rotate(frame.Q, frame.u0)
        Qalign = _rotation(axis, TF(π))
    else
        axis = cross(current, target)
        Qalign = Quaternion{TF}(one(TF) + d, axis[1], axis[2], axis[3])
        Qalign /= norm(Qalign)
    end

    frame.Q = Qalign * frame.Q
    frame.Q /= norm(frame.Q)

    return nothing
end

"""
    _rotation(axis::SVector{3, TF}, angle::TF) where {TF <: AbstractFloat}

Construct a quaternion representing a rotation about `axis` by `angle`.

# Parameters
- `axis`: Rotation axis.
- `angle`: Rotation angle.
"""
@inline function _rotation(axis :: SVector{3,TF}, angle :: TF) :: Quaternion{TF} where {TF <: AbstractFloat}
    na = norm(axis)
    iszero(na) && throw(ArgumentError("Rotation axis must be nonzero"))

    n = axis / na
    halfangle = angle * TF(0.5)

    c = cos(halfangle)
    s = sin(halfangle)
    sn = s * n

    return Quaternion{TF}(c, sn[1], sn[2], sn[3])
end

"""
    _rotate(Q::Quaternion{TF}, v::SVector{3, TF}) where {TF <: AbstractFloat}

Apply the rotation represented by `Q` to the vector `v`.

# Parameters
- `Q`: Quaternion representing the rotation.
- `v`: Vector to rotate.
"""
@inline function _rotate(Q :: Quaternion{TF}, v :: SVector{3,TF}) where {TF<:AbstractFloat}
    qr = Q * Quaternion{TF}(zero(TF), v...) * conj(Q)
    return @SVector [qr.v1, qr.v2, qr.v3]
end

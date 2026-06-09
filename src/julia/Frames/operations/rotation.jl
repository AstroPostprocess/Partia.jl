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

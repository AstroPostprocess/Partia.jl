

"""
    Frame{TF <: AbstractFloat}

Mutable reference frame storing a position, an initial local basis, and the current orientation quaternion.

# Fields
- `x`: Current frame position in global coordinates.
- `f0`: Initial forward direction.
- `u0`: Initial up direction.
- `r0`: Initial right direction derived from the initial forward and up directions.
- `Q`: Current orientation quaternion.
"""
mutable struct Frame{TF <: AbstractFloat}
    # Position of Frame
    x   :: MVector{3, TF}

    # Initial local basis
    f0  :: SVector{3, TF} 
    u0  :: SVector{3, TF} 
    r0  :: SVector{3, TF} 

    # Quaternion
    Q   :: Quaternion{TF}

end


"""
    Frame(x::NTuple{3, TF}, f0::NTuple{3, TF}, u0::NTuple{3, TF}) where {TF <: AbstractFloat}

Construct a `Frame` from a global position and explicit initial forward and up directions.
The initial forward and up directions are normalized and must be nonzero and orthogonal.
The initial right direction is derived from the normalized forward and up directions, and the orientation quaternion starts as the identity rotation.

# Parameters
- `x`: Initial frame position in global coordinates.
- `f0`: Initial forward direction.
- `u0`: Initial up direction.
"""
function Frame(x :: NTuple{3, TF}, f0 :: NTuple{3, TF}, u0 :: NTuple{3, TF}) where {TF <: AbstractFloat}
    # normalization
    nf0 = hypot(f0...)
    nu0 = hypot(u0...)
    
    # Check whether there has zero vector
    iszero(nf0) && throw(ArgumentError("f0 must be nonzero."))
    iszero(nu0) && throw(ArgumentError("u0 must be nonzero."))

    invnf0  = inv(nf0)  
    invnu0  = inv(nu0)  

    xvec     = MVector{3, TF}(x)
    f0normed = SVector{3, TF}(f0) * invnf0
    u0normed = SVector{3, TF}(u0) * invnu0
    

    # Check whether orthonormal
    tol = sqrt(eps(TF))

    abs(dot(f0normed, u0normed)) > tol && throw(ArgumentError("f0 and u0 must be orthogonal."))

    # Construct r with the identity: r = f × u
    r0 = cross(f0normed, u0normed)
    r0normed = r0 / norm(r0)

    # Construct the initial Quaternions
    Q = Quaternion{TF}(one(TF), zero(TF), zero(TF), zero(TF))

    return Frame{TF}(
        xvec,
        f0normed,
        u0normed,
        r0normed,
        Q
    )
end

"""
    Frame(x::TF, y::TF, z::TF) where {TF <: AbstractFloat}

Construct a `Frame` at the global position `(x, y, z)` with its initial forward direction pointing toward the origin.
The initial up direction is chosen tangent to the sphere through the position and oriented toward the polar direction, with a pole-safe fallback.
The initial right direction is derived from the initial forward and up directions, and the orientation quaternion starts as the identity rotation.

# Parameters
- `x`: Initial global x coordinate.
- `y`: Initial global y coordinate.
- `z`: Initial global z coordinate.
"""
function Frame(x :: TF, y :: TF, z :: TF) where {TF <: AbstractFloat}
    # line of sight would initial set towards the origin
    ρ = hypot(x, y, z)
    iszero(ρ) && throw(ArgumentError("Frame cannot be created through this dispatch, please use the standard constructor!"))
    invρ = inv(ρ)
    f0x = -x * invρ
    f0y = -y * invρ
    f0z = -z * invρ

    # At that position, u0 is the tangent vector on a sphere with r² = x² + y² + z², towards the pole on θ=0
    s = hypot(x, y)
    if s ≤ sqrt(eps(TF)) * ρ
        # Make right vector is x
        u0x = zero(TF)
        u0y = ifelse(z > zero(TF), one(TF), -one(TF))
        u0z = zero(TF)
    else
        invs = inv(s)
        invsρ = invs * invρ

        u0x = - invsρ * x * z
        u0y = - invsρ * y * z
        u0z =   invsρ * s * s
    end
    
    # Construct r with the identity: r = f × u
    f0normed = @SVector TF[f0x, f0y, f0z]
    u0normed = @SVector TF[u0x, u0y, u0z]

    r0 = cross(f0normed, u0normed)
    r0normed = r0 / norm(r0)

    # Construct the initial Quaternions
    Q = Quaternion{TF}(one(TF), zero(TF), zero(TF), zero(TF))

    return Frame{TF}(
        MVector{3, TF}(x, y, z),
        f0normed,
        u0normed,
        r0normed,
        Q
    )
end


"""
    frame_position(frame::Frame{TF}) where {TF <: AbstractFloat}

Return the current frame position as a tuple in global coordinates.

# Parameters
- `frame`: Frame whose current position is returned.
"""
@inline function frame_position(frame :: Frame{TF}) where {TF <: AbstractFloat}
    return Tuple(frame.x)
end

"""
    frame_right(frame::Frame{TF}) where {TF <: AbstractFloat}

Return the frame's current right direction as a tuple in global coordinates.
The current right direction is obtained by rotating `frame.r0` using `frame.Q`.

# Parameters
- `frame`: Frame whose current right direction is returned.
"""
@inline function frame_right(frame :: Frame{TF}) where {TF <: AbstractFloat}
    return Tuple(_rotate(frame.Q, frame.r0))
end

"""
    frame_forward(frame::Frame{TF}) where {TF <: AbstractFloat}

Return the frame's current forward direction as a tuple in global coordinates.
The current forward direction is obtained by rotating `frame.f0` using `frame.Q`.

# Parameters
- `frame`: Frame whose current forward direction is returned.
"""
@inline function frame_forward(frame :: Frame{TF}) where {TF <: AbstractFloat}
    return Tuple(_rotate(frame.Q, frame.f0))
end

"""
    frame_up(frame::Frame{TF}) where {TF <: AbstractFloat}

Return the frame's current up direction as a tuple in global coordinates.
The current up direction is obtained by rotating `frame.u0` using `frame.Q`.

# Parameters
- `frame`: Frame whose current up direction is returned.
"""
@inline function frame_up(frame :: Frame{TF}) where {TF <: AbstractFloat}
    return Tuple(_rotate(frame.Q, frame.u0))
end

"""
    frame_basis(frame::Frame{TF}) where {TF <: AbstractFloat}

Return the frame's current local basis directions as a named tuple in global coordinates.
The current right, forward, and up directions are obtained by rotating `frame.r0`, `frame.f0`, and `frame.u0` using `frame.Q`.

# Parameters
- `frame`: Frame whose current local basis directions are returned.
"""
@inline function frame_basis(frame :: Frame{TF}) where {TF <: AbstractFloat}
    Q = frame.Q

    r = Tuple(_rotate(Q, frame.r0))
    f = Tuple(_rotate(Q, frame.f0))
    u = Tuple(_rotate(Q, frame.u0))

    return (right=r, forward=f, up=u)
end

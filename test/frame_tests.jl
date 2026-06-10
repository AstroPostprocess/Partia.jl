######################################################################################

#  Test: Frames -- Orientation, Translation, Accessors, and Constructors
#  What this file tests
#  Focused validation of the `Partia.Frames` reference-frame implementation:
#  1. Quaternion helpers
#     -- Identity rotations, known axis-angle rotations, axis invariance,
#        norm preservation, unit-quaternion construction, and invalid axes.
#     -- Independent Rodrigues-formula checks for `_rotation` / `_rotate`.
#     -- Quaternion composition order, including non-commutativity.
#  2. Frame rotation
#     -- `rotate!` quaternion accumulation and an independent local-axis
#        yaw-pitch-roll semantics check based only on Rodrigues rotation.
#     -- Quaternion normalization, current-basis orthonormality, and handedness.
#  3. Translation
#     -- Global-coordinate and local-coordinate translation, including
#        single-axis local movement and global-vs-local behavior.
#  4. Accessors and constructors
#     -- Position and basis accessors, non-aliasing, explicit-basis
#        constructor invariants, look-at-origin construction, pole handling,
#        and error paths.

######################################################################################
using Test
using LinearAlgebra
using StaticArrays
using Quaternions
using Partia

# ========================== Public API imports ============================== #

import Partia: rotate_forward_to!, translate!,
    frame_position, frame_right, frame_forward, frame_up, frame_basis

# ========================== Module aliases ================================== #

const FrameAPI = Partia.Frames

# ========================== Helper functions ================================ #

frame_tol(::Type{TF}) where {TF <: AbstractFloat} = TF(256) * eps(TF)

sv3(x) = SVector{3}(x)
v3(::Type{TF}, x, y, z) where {TF <: AbstractFloat} = SVector{3, TF}(TF(x), TF(y), TF(z))

function rodrigues_rotate(v :: SVector{3, TF}, axis :: SVector{3, TF}, angle :: TF) where {TF <: AbstractFloat}
    n = axis / norm(axis)
    return (
        v * cos(angle) +
        cross(n, v) * sin(angle) +
        n * dot(n, v) * (one(TF) - cos(angle))
    )
end

qcomponents(Q) = SVector{4}(Q.s, Q.v1, Q.v2, Q.v3)

function qdist(Q1, Q2)
    q1 = qcomponents(Q1)
    q2 = qcomponents(Q2)

    return min(
        norm(q1 - q2),
        norm(q1 + q2),
    )
end

vdist(v1, v2) = norm(sv3(v1) - sv3(v2))

# ============================== Test body =================================== #

# ── 1. Quaternion rotation primitives ────────────────────────────────── #

@testset "Frames -- quaternion primitives" begin
    for TF in (Float32, Float64)
        tol = frame_tol(TF)
        identity_Q = Quaternion{TF}(one(TF), zero(TF), zero(TF), zero(TF))

        @testset "identity rotation $(TF)" begin
            vectors = (
                v3(TF, 1, 0, 0),
                v3(TF, 0, 1, 0),
                v3(TF, 0, 0, 1),
                v3(TF, 0.25, -1.5, 2.0),
            )

            for v in vectors
                @test vdist(FrameAPI._rotate(identity_Q, v), v) <= tol
            end
        end

        @testset "known axis-angle rotations $(TF)" begin
            Qz = FrameAPI._rotation(v3(TF, 0, 0, 1), TF(pi / 2))
            @test vdist(FrameAPI._rotate(Qz, v3(TF, 1, 0, 0)), v3(TF, 0, 1, 0)) <= tol

            Qx = FrameAPI._rotation(v3(TF, 1, 0, 0), TF(pi / 2))
            @test vdist(FrameAPI._rotate(Qx, v3(TF, 0, 1, 0)), v3(TF, 0, 0, 1)) <= tol
        end

        @testset "axis invariance, norm preservation, and unit quaternion $(TF)" begin
            axis = normalize(v3(TF, 2, -3, 4))
            angle = TF(0.73)
            Q = FrameAPI._rotation(axis, angle)

            @test vdist(FrameAPI._rotate(Q, axis), axis) <= tol
            @test abs(norm(Q) - one(TF)) <= tol

            axes = (
                v3(TF, 1, 2, -1),
                v3(TF, -3, 1, 2),
                v3(TF, 2, 3, 5),
            )
            vectors = (
                v3(TF, 0.25, -0.75, 1.5),
                v3(TF, -2, 0.5, 0.125),
                v3(TF, 1.25, 2.5, -0.5),
            )
            angles = (TF(0.37), TF(-1.1), TF(2.35))

            for (axis, v, angle) in zip(axes, vectors, angles)
                Q = FrameAPI._rotation(axis, angle)
                @test abs(norm(FrameAPI._rotate(Q, v)) - norm(v)) <= tol
            end
        end

        @testset "Rodrigues validation $(TF)" begin
            cases = (
                (v3(TF, 1.2, -0.4, 2.5), v3(TF, 0.7, 1.1, -0.3), TF(0.41)),
                (v3(TF, -1.0, 0.8, 0.6), v3(TF, 1.0, -2.0, 0.5), TF(-0.92)),
                (v3(TF, 0.3, 1.7, -1.1), v3(TF, -0.4, 0.9, 1.3), TF(1.37)),
            )

            for (v, axis, angle) in cases
                Q = FrameAPI._rotation(axis, angle)
                expected = rodrigues_rotate(v, axis, angle)
                @test vdist(FrameAPI._rotate(Q, v), expected) <= tol
            end
        end

        @testset "quaternion composition order $(TF)" begin
            Q1 = FrameAPI._rotation(v3(TF, 1, 2, 0), TF(0.64))
            Q2 = FrameAPI._rotation(v3(TF, -1, 0, 3), TF(-0.83))
            v = v3(TF, 0.5, -1.25, 2.0)

            composed = FrameAPI._rotate(Q2 * Q1, v)
            sequential = FrameAPI._rotate(Q2, FrameAPI._rotate(Q1, v))
            reversed = FrameAPI._rotate(Q1 * Q2, v)

            @test vdist(composed, sequential) <= tol
            @test vdist(composed, reversed) > TF(10) * tol
        end
    end

    @test_throws ArgumentError FrameAPI._rotation(SVector{3, Float64}(0.0, 0.0, 0.0), 0.5)
end

# ── 2. rotate! orientation updates and current basis ─────────────────── #

@testset "Frames -- rotate! orientation and basis" begin
    for TF in (Float32, Float64)
        tol = frame_tol(TF)

        @testset "quaternion accumulation order $(TF)" begin
            frame = Frame((TF(1), TF(2), TF(3)), (TF(0), TF(0), TF(-1)), (TF(0), TF(1), TF(0)))
            yaw, pitch, roll = TF(0.43), TF(-0.31), TF(0.27)

            Q = Quaternion{TF}(one(TF), zero(TF), zero(TF), zero(TF))
            Qyaw = FrameAPI._rotation(frame.u0, yaw)
            Qpitch = FrameAPI._rotation(frame.r0, pitch)
            Qroll = FrameAPI._rotation(frame.f0, roll)
            Q = Q * Qyaw
            Q = Q * Qpitch
            Q = Q * Qroll
            Q /= norm(Q)

            rotate!(frame, yaw, pitch, roll)

            @test qdist(frame.Q, Q) <= tol
            @test vdist(sv3(frame_right(frame)), FrameAPI._rotate(Q, frame.r0)) <= tol
            @test vdist(sv3(frame_forward(frame)), FrameAPI._rotate(Q, frame.f0)) <= tol
            @test vdist(sv3(frame_up(frame)), FrameAPI._rotate(Q, frame.u0)) <= tol
        end

        @testset "independent local-axis semantics $(TF)" begin
            frame = Frame((TF(1.5), TF(-2.0), TF(0.75)), (TF(1), TF(2), TF(-2)), (TF(2), TF(1), TF(2)))
            yaw, pitch, roll = TF(0.38), TF(-0.47), TF(0.29)
            p0 = frame_position(frame)

            r = frame.r0
            f = frame.f0
            u = frame.u0

            r = rodrigues_rotate(r, u, yaw)
            f = rodrigues_rotate(f, u, yaw)

            f = rodrigues_rotate(f, r, pitch)
            u = rodrigues_rotate(u, r, pitch)

            r = rodrigues_rotate(r, f, roll)
            u = rodrigues_rotate(u, f, roll)

            rotate!(frame, yaw, pitch, roll)

            @test frame_position(frame) == p0
            @test vdist(frame_right(frame), r) <= tol
            @test vdist(frame_forward(frame), f) <= tol
            @test vdist(frame_up(frame), u) <= tol
        end

        @testset "accumulated quaternion normalization and orthonormal basis $(TF)" begin
            frame = Frame((TF(2), TF(-1), TF(3)), (TF(0), TF(0), TF(-1)), (TF(0), TF(1), TF(0)))

            for angles in ((TF(0.2), TF(-0.4), TF(0.1)),
                           (TF(-0.3), TF(0.25), TF(0.5)),
                           (TF(0.7), TF(-0.2), TF(-0.35)))
                rotate!(frame, angles...)
                @test abs(norm(frame.Q) - one(TF)) <= tol
            end

            b = frame_basis(frame)
            r = sv3(b.right)
            f = sv3(b.forward)
            u = sv3(b.up)

            @test abs(norm(r) - one(TF)) <= tol
            @test abs(norm(f) - one(TF)) <= tol
            @test abs(norm(u) - one(TF)) <= tol
            @test abs(dot(r, f)) <= tol
            @test abs(dot(r, u)) <= tol
            @test abs(dot(f, u)) <= tol
            @test vdist(cross(f, u), r) <= tol
        end
    end
end

# ── 3. Forward-direction alignment ───────────────────────────────────── #

@testset "Frames -- rotate_forward_to!" begin
    for TF in (Float32, Float64)
        tol = frame_tol(TF)

        @testset "aligns current forward to target $(TF)" begin
            frame = Frame((TF(1), TF(2), TF(3)), (TF(0), TF(0), TF(-1)), (TF(0), TF(1), TF(0)))
            p0 = frame_position(frame)
            target = (TF(1), TF(2), TF(-3))
            target_normed = sv3(target) / norm(sv3(target))

            rotate_forward_to!(frame, target)

            @test frame_position(frame) == p0
            @test vdist(frame_forward(frame), target_normed) <= tol
            @test abs(norm(frame.Q) - one(TF)) <= tol
        end

        @testset "composes from an existing orientation $(TF)" begin
            frame = Frame((TF(-2), TF(0.5), TF(1)), (TF(0), TF(0), TF(-1)), (TF(0), TF(1), TF(0)))
            rotate!(frame, TF(0.4), TF(-0.25), TF(0.35))

            target = SVector{3,TF}(TF(-0.3), TF(0.7), TF(-0.2))
            target /= norm(target)
            rotate_forward_to!(frame, target)

            b = frame_basis(frame)
            r = sv3(b.right)
            f = sv3(b.forward)
            u = sv3(b.up)

            @test vdist(f, target) <= tol
            @test abs(norm(r) - one(TF)) <= tol
            @test abs(norm(f) - one(TF)) <= tol
            @test abs(norm(u) - one(TF)) <= tol
            @test abs(dot(r, f)) <= tol
            @test abs(dot(f, u)) <= tol
            @test vdist(cross(f, u), r) <= tol
        end

        @testset "opposite direction and invalid target $(TF)" begin
            frame = Frame((TF(0), TF(0), TF(0)), (TF(0), TF(0), TF(-1)), (TF(0), TF(1), TF(0)))

            rotate_forward_to!(frame, (TF(0), TF(0), TF(1)))
            @test vdist(frame_forward(frame), v3(TF, 0, 0, 1)) <= tol

            @test_throws ArgumentError rotate_forward_to!(frame, (TF(0), TF(0), TF(0)))
            @test_throws DimensionMismatch rotate_forward_to!(frame, TF[1, 0])
        end
    end
end

# ── 3. Global and local translation ──────────────────────────────────── #

@testset "Frames -- translation" begin
    for TF in (Float32, Float64)
        tol = frame_tol(TF)

        @testset "global translation $(TF)" begin
            frame = Frame((TF(1), TF(-2), TF(3)), (TF(0), TF(0), TF(-1)), (TF(0), TF(1), TF(0)))
            rotate!(frame, TF(0.4), TF(-0.2), TF(0.3))
            Q0 = frame.Q

            translate!(GlobalCoordinates, frame, TF(1.5), TF(-0.25), TF(2.25))

            @test vdist(frame_position(frame), v3(TF, 2.5, -2.25, 5.25)) <= tol
            @test qdist(frame.Q, Q0) <= tol
        end

        @testset "local translation $(TF)" begin
            frame = Frame((TF(-1), TF(0.5), TF(2)), (TF(0), TF(0), TF(-1)), (TF(0), TF(1), TF(0)))
            rotate!(frame, TF(0.55), TF(-0.35), TF(0.2))
            start = sv3(frame_position(frame))
            Q0 = frame.Q
            r = sv3(frame_right(frame))
            f = sv3(frame_forward(frame))
            u = sv3(frame_up(frame))
            Δr, Δf, Δu = TF(0.75), TF(-1.25), TF(0.5)

            translate!(LocalCoordinates, frame, Δr, Δf, Δu)

            @test vdist(frame_position(frame), start + Δr * r + Δf * f + Δu * u) <= tol
            @test qdist(frame.Q, Q0) <= tol
        end

        @testset "local single-axis translation $(TF)" begin
            displacements = (
                (TF(1.25), TF(0), TF(0), :right),
                (TF(0), TF(-0.75), TF(0), :forward),
                (TF(0), TF(0), TF(1.5), :up),
            )

            for (Δr, Δf, Δu, axis_name) in displacements
                frame = Frame((TF(0.5), TF(-1.0), TF(1.5)), (TF(0), TF(0), TF(-1)), (TF(0), TF(1), TF(0)))
                rotate!(frame, TF(0.3), TF(0.4), TF(-0.2))
                start = sv3(frame_position(frame))
                r = sv3(frame_right(frame))
                f = sv3(frame_forward(frame))
                u = sv3(frame_up(frame))

                translate!(LocalCoordinates, frame, Δr, Δf, Δu)

                expected = axis_name === :right ? start + Δr * r :
                           axis_name === :forward ? start + Δf * f :
                           start + Δu * u
                @test vdist(frame_position(frame), expected) <= tol
            end
        end

        @testset "global versus local translation $(TF)" begin
            global_frame = Frame((TF(0), TF(0), TF(0)), (TF(0), TF(0), TF(-1)), (TF(0), TF(1), TF(0)))
            local_frame = Frame((TF(0), TF(0), TF(0)), (TF(0), TF(0), TF(-1)), (TF(0), TF(1), TF(0)))
            rotate!(global_frame, TF(pi / 2), zero(TF), zero(TF))
            rotate!(local_frame, TF(pi / 2), zero(TF), zero(TF))

            translate!(GlobalCoordinates, global_frame, TF(1), TF(0), TF(0))
            translate!(LocalCoordinates, local_frame, TF(1), TF(0), TF(0))

            @test vdist(frame_position(global_frame), v3(TF, 1, 0, 0)) <= tol
            @test vdist(frame_position(local_frame), sv3(frame_right(local_frame))) <= tol
            @test vdist(frame_position(global_frame), frame_position(local_frame)) > TF(10) * tol
        end
    end
end

# ── 4. Accessor behavior ─────────────────────────────────────────────── #

@testset "Frames -- accessors" begin
    for TF in (Float32, Float64)
        tol = frame_tol(TF)
        frame = Frame((TF(1), TF(2), TF(3)), (TF(0), TF(0), TF(-1)), (TF(0), TF(1), TF(0)))
        rotate!(frame, TF(0.45), TF(-0.25), TF(0.6))

        @test frame_position(frame) == (TF(1), TF(2), TF(3))
        @test vdist(frame_right(frame), FrameAPI._rotate(frame.Q, frame.r0)) <= tol
        @test vdist(frame_forward(frame), FrameAPI._rotate(frame.Q, frame.f0)) <= tol
        @test vdist(frame_up(frame), FrameAPI._rotate(frame.Q, frame.u0)) <= tol

        b = frame_basis(frame)
        @test b isa NamedTuple{(:right, :forward, :up)}
        @test b.right == frame_right(frame)
        @test b.forward == frame_forward(frame)
        @test b.up == frame_up(frame)

        p = frame_position(frame)
        frame.x[1] += TF(10)
        @test p == (TF(1), TF(2), TF(3))
    end
end

# ── 5. Constructor invariants and error paths ────────────────────────── #

@testset "Frames -- constructors" begin
    for TF in (Float32, Float64)
        tol = frame_tol(TF)

        @testset "explicit initial basis $(TF)" begin
            frame = Frame((TF(1), TF(2), TF(3)), (TF(0), TF(0), TF(-2)), (TF(0), TF(5), TF(0)))

            @test abs(norm(frame.f0) - one(TF)) <= tol
            @test abs(norm(frame.u0) - one(TF)) <= tol
            @test abs(dot(frame.f0, frame.u0)) <= tol
            @test vdist(frame.r0, cross(frame.f0, frame.u0)) <= tol
            @test abs(norm(frame.r0) - one(TF)) <= tol
            @test qdist(frame.Q, Quaternion{TF}(one(TF), zero(TF), zero(TF), zero(TF))) <= tol
            @test vdist(frame_right(frame), frame.r0) <= tol
            @test vdist(frame_forward(frame), frame.f0) <= tol
            @test vdist(frame_up(frame), frame.u0) <= tol
        end

        @testset "invalid explicit basis $(TF)" begin
            @test_throws ArgumentError Frame((TF(0), TF(0), TF(0)), (zero(TF), zero(TF), zero(TF)), (zero(TF), one(TF), zero(TF)))
            @test_throws ArgumentError Frame((TF(0), TF(0), TF(0)), (zero(TF), zero(TF), -one(TF)), (zero(TF), zero(TF), zero(TF)))
            @test_throws ArgumentError Frame((TF(0), TF(0), TF(0)), (one(TF), zero(TF), zero(TF)), (one(TF), one(TF), zero(TF)))
        end

        @testset "look-at-origin constructor $(TF)" begin
            positions = (
                (TF(3), TF(4), TF(12)),
                (TF(0), TF(0), TF(2)),
                (TF(0), TF(0), TF(-2)),
                (sqrt(eps(TF)) * TF(0.25), zero(TF), one(TF)),
            )

            for pos in positions
                frame = Frame(pos...)
                p = v3(TF, pos[1], pos[2], pos[3])
                expected_f = -p / norm(p)
                r, f, u = frame.r0, frame.f0, frame.u0

                @test vdist(f, expected_f) <= tol
                @test abs(norm(r) - one(TF)) <= tol
                @test abs(norm(f) - one(TF)) <= tol
                @test abs(norm(u) - one(TF)) <= tol
                @test abs(dot(r, f)) <= tol
                @test abs(dot(r, u)) <= tol
                @test abs(dot(f, u)) <= tol
                @test vdist(cross(f, u), r) <= tol
                @test all(isfinite, r)
                @test all(isfinite, f)
                @test all(isfinite, u)

                if pos == positions[1]
                    radial = p / norm(p)
                    global_up = v3(TF, 0, 0, 1)
                    expected_u_raw = global_up - dot(global_up, radial) * radial
                    expected_u = expected_u_raw / norm(expected_u_raw)

                    @test vdist(frame.u0, expected_u) <= tol
                end
            end

            @test vdist(Frame(TF(0), TF(0), TF(2)).r0, v3(TF, 1, 0, 0)) <= tol
            @test vdist(Frame(TF(0), TF(0), TF(-2)).r0, v3(TF, 1, 0, 0)) <= tol
            @test_throws ArgumentError Frame(zero(TF), zero(TF), zero(TF))
        end
    end
end

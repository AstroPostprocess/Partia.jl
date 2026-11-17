# Julia squared-distance-only port of David Eberly's 3D query.
# Copyright (c) 1998-2026 David Eberly, Geometric Tools, Redmond WA 98052.
# Distributed under the Boost Software License, Version 1.0 (see LICENSE_EBERLY.txt).
# Source: https://github.com/davideberly/GeometricTools/blob/master/GTE/Mathematics/DistLine3CanonicalBox3.h
# File Version: 8.0.2025.05.10. Retrieved 2026-11-18.
# Retains Face/DoQuery3D/DoQuery2D branch decisions. Repeated edge branches
# are factored into eberly_edge; closest-point/parameter outputs are omitted.
# All arithmetic, including center/extent conversion, uses the input type.

@inline function eberly_edge(i0, i1, i2, p, v, e, m, q, len2, tmp)
    if tmp <= 2 * len2 * e[i1]
        t = tmp / len2
        w = q[i1] - t
        len2 += v[i1]^2
        delta = v[i0]*m[i0] + v[i1]*w + v[i2]*q[i2]
        return m[i0]^2 + w^2 + q[i2]^2 - delta^2 / len2
    else
        len2 += v[i1]^2
        delta = v[i0]*m[i0] + v[i1]*m[i1] + v[i2]*q[i2]
        return m[i0]^2 + m[i1]^2 + q[i2]^2 - delta^2 / len2
    end
end

@inline function eberly_face(i0, i1, i2, p, v, e, m)
    q = p .+ e
    if v[i0]*q[i1] >= v[i1]*m[i0]
        if v[i0]*q[i2] >= v[i2]*m[i0]
            return zero(eltype(p))
        end
        len2 = v[i0]^2 + v[i2]^2
        tmp = len2*q[i1] - v[i1]*(v[i0]*m[i0] + v[i2]*q[i2])
        return eberly_edge(i0, i1, i2, p, v, e, m, q, len2, tmp)
    elseif v[i0]*q[i2] >= v[i2]*m[i0]
        len2 = v[i0]^2 + v[i1]^2
        tmp = len2*q[i2] - v[i2]*(v[i0]*m[i0] + v[i1]*q[i1])
        return eberly_edge(i0, i2, i1, p, v, e, m, q, len2, tmp)
    else
        len2 = v[i0]^2 + v[i2]^2
        tmp = len2*q[i1] - v[i1]*(v[i0]*m[i0] + v[i2]*q[i2])
        if tmp >= 0
            return eberly_edge(i0, i1, i2, p, v, e, m, q, len2, tmp)
        end
        len2 = v[i0]^2 + v[i1]^2
        tmp = len2*q[i2] - v[i2]*(v[i0]*m[i0] + v[i1]*q[i1])
        if tmp >= 0
            return eberly_edge(i0, i2, i1, p, v, e, m, q, len2, tmp)
        end
        len2 += v[i2]^2
        delta = v[i0]*m[i0] + v[i1]*q[i1] + v[i2]*q[i2]
        return m[i0]^2 + q[i1]^2 + q[i2]^2 - delta^2 / len2
    end
end

@inline function eberly_2d(i0, i1, i2, p, v, e)
    prod0 = v[i1]*(p[i0]-e[i0])
    prod1 = v[i0]*(p[i1]-e[i1])
    delta = prod0 >= prod1 ? prod0-v[i0]*(p[i1]+e[i1]) :
                                   prod1-v[i1]*(p[i0]+e[i0])
    d2 = delta >= 0 ? delta^2/(v[i0]^2+v[i1]^2) : zero(eltype(p))
    return d2 + max(abs(p[i2])-e[i2], zero(eltype(p)))^2
end

@inline function eberly_line_aabb(origin::NTuple{3,T}, direction::NTuple{3,T},
                                  aabb_min, aabb_max, idx::Int) where {T}
    c = ntuple(d -> (aabb_min[d][idx]+aabb_max[d][idx])/T(2), 3)
    e = ntuple(d -> (aabb_max[d][idx]-aabb_min[d][idx])/T(2), 3)
    p = ntuple(d -> direction[d] < 0 ? c[d]-origin[d] : origin[d]-c[d], 3)
    v = abs.(direction)
    d2 = if v[1] > 0 && v[2] > 0 && v[3] > 0
        m = p .- e
        if v[2]*m[1] >= v[1]*m[2]
            v[3]*m[1] >= v[1]*m[3] ? eberly_face(1,2,3,p,v,e,m) : eberly_face(3,1,2,p,v,e,m)
        else
            v[3]*m[2] >= v[2]*m[3] ? eberly_face(2,3,1,p,v,e,m) : eberly_face(3,1,2,p,v,e,m)
        end
    elseif v[1] > 0 && v[2] > 0
        eberly_2d(1,2,3,p,v,e)
    elseif v[1] > 0 && v[3] > 0
        eberly_2d(1,3,2,p,v,e)
    elseif v[2] > 0 && v[3] > 0
        eberly_2d(2,3,1,p,v,e)
    else
        sum(ntuple(d -> iszero(v[d]) ? max(abs(p[d])-e[d],zero(T))^2 : zero(T), 3))
    end
    # The original subtractive formula can round slightly below zero.
    return max(d2, zero(T))
end

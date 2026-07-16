######################################################################################

# Axis-aligned bounding box data structure.
#     by Wei-Shan Su,
#     May 4, 2026

######################################################################################

"""
    AABB{D, TF, VF}

Structure-of-arrays storage for axis-aligned bounding-box minima and maxima.

# Fields
- `min :: NTuple{D, VF}`: Per-axis lower bounds.
- `max :: NTuple{D, VF}`: Per-axis upper bounds.
"""
struct AABB{D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
    min :: NTuple{D, VF}
    max :: NTuple{D, VF}
end

function Adapt.adapt_structure(to, x :: AB) where {D, AB <: AABB{D}}
    AABB(
        ntuple(i -> Adapt.adapt(to, x.min[i]), D),
        ntuple(i -> Adapt.adapt(to, x.max[i]), D)
    )
end

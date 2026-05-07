######################################################################################

# Axis-aligned bounding box data structure.
#     by Wei-Shan Su,
#     May 4, 2026

######################################################################################

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

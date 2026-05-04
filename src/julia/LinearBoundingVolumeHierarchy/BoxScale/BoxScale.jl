######################################################################################

# Box scale data structure for describing per-particle spatial extents.
#     by Wei-Shan Su,
#     May 4, 2026

######################################################################################
struct BoxScale{TF <: AbstractFloat, VF <: AbstractVector{TF}}
    scale :: VF
    is_morton_sorted :: Bool
end

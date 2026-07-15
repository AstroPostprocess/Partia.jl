"""
    GridDataset_params_TYPE(T::Type{<:AbstractFloat})

Return the concrete metadata dictionary type used by `GridDataset` for
floating-point type `T`.

# Parameters
- `T`: Floating-point type permitted in metadata values.

# Returns
- `Type{<:Dict}`: Dictionary type with `Symbol` keys and supported metadata
  value types.
"""
@inline GridDataset_params_TYPE( :: Type{T}) where {T <: AbstractFloat} = Dict{Symbol, Union{String, Int, Bool, T}}

"""
    GridDataset{L, TF, G}

Serializable grid bundle together with operation and unit metadata.

# Fields
- `data :: GridBundle{L, G}`: Named grid data.
- `params`: Metadata dictionary with symbolic keys.
"""
struct GridDataset{L, TF <: AbstractFloat, G <: AbstractGrid{TF}}
    data :: GridBundle{L,G}
    params :: Dict{Symbol, Union{String, Int, Bool, TF}}
end

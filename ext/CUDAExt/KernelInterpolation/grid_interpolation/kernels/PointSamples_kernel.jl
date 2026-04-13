@inline function _point_samples_interpolation_kernel!(grids :: NTuple{L, PointSamples{3, TF}}, input :: InterpolationInput{3, TF}, catalog_consice :: InterpolationCatalogConcise{3, N, G, Div, C}, LBVH :: LinearBVH, itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) where {N, G, Div, C, L, TF <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    tid    = Int(CUDA.threadIdx().x)
    bid    = Int(CUDA.blockIdx().x)
    bdim   = Int(CUDA.blockDim().x)
    gdim   = Int(CUDA.gridDim().x)

    gid    = (bid - 1) * bdim + tid
    stride = bdim * gdim

    npoints = length(grids[1])
    i = gid
    while i <= npoints
        @inbounds begin
            xa = grids[1].coor[1][i]
            ya = grids[1].coor[2][i]
            za = grids[1].coor[3][i]
            point :: NTuple{3, TF} = (xa, ya, za)
        end

        ha = LBVH_find_nearest_h(LBVH, point)

        itpresult :: Tuple{NTuple{N,TF}, NTuple{G,NTuple{3,TF}}, NTuple{Div,TF}, NTuple{C,NTuple{3,TF}}} = Partia.KernelInterpolation._general_quantity_interpolate_kernel(input, point, ha, LBVH, catalog_consice, itp_strategy)

        scalars :: NTuple{N,TF} = itpresult[1]
        gradients :: NTuple{G,NTuple{3,TF}} = itpresult[2]
        divergences :: NTuple{Div,TF} = itpresult[3]
        curls :: NTuple{C,NTuple{3,TF}} = itpresult[4]

        out_idx = 1

        if N > 0
            @inbounds for j in 1:N
                grids[out_idx].grid[i] = scalars[j]
                out_idx += 1
            end
        end

        if G > 0
            @inbounds for j in 1:G
                grad_quant = gradients[j]
                @inbounds for d in 1:3
                    grids[out_idx].grid[i] = grad_quant[d]
                    out_idx += 1
                end
            end
        end

        if Div > 0
            @inbounds for j in 1:Div
                div_quant = divergences[j]
                grids[out_idx].grid[i] = div_quant
                out_idx += 1
            end
        end

        if C > 0
            @inbounds for j in 1:C
                curl_quant = curls[j]
                @inbounds for d in 1:3
                    grids[out_idx].grid[i] = curl_quant[d]
                    out_idx += 1
                end
            end
        end
        i += stride
    end
    return nothing
end

@inline function _point_samples_interpolation_kernel!(grids :: NTuple{L, PointSamples{3, TF}}, input :: InterpolationInput{3, TF}, catalog_consice :: InterpolationCatalogConcise{3, N, G, Div, C}, LBVH :: LinearBVH, ::Type{itpScatter}) where {N, G, Div, C, L, TF <: AbstractFloat}
    tid    = Int(CUDA.threadIdx().x)
    bid    = Int(CUDA.blockIdx().x)
    bdim   = Int(CUDA.blockDim().x)
    gdim   = Int(CUDA.gridDim().x)

    gid    = (bid - 1) * bdim + tid
    stride = bdim * gdim

    npoints = length(grids[1])
    i = gid
    while i <= npoints
        @inbounds begin
            xa = grids[1].coor[1][i]
            ya = grids[1].coor[2][i]
            za = grids[1].coor[3][i]
            point :: NTuple{3, TF} = (xa, ya, za)
        end

        itpresult :: Tuple{NTuple{N,TF}, NTuple{G,NTuple{3,TF}}, NTuple{Div,TF}, NTuple{C,NTuple{3,TF}}} = Partia.KernelInterpolation._general_quantity_interpolate_kernel(input, point, LBVH, catalog_consice, itpScatter)

        scalars :: NTuple{N,TF} = itpresult[1]
        gradients :: NTuple{G,NTuple{3,TF}} = itpresult[2]
        divergences :: NTuple{Div,TF} = itpresult[3]
        curls :: NTuple{C,NTuple{3,TF}} = itpresult[4]

        out_idx = 1

        if N > 0
            @inbounds for j in 1:N
                grids[out_idx].grid[i] = scalars[j]
                out_idx += 1
            end
        end

        if G > 0
            @inbounds for j in 1:G
                grad_quant = gradients[j]
                @inbounds for d in 1:3
                    grids[out_idx].grid[i] = grad_quant[d]
                    out_idx += 1
                end
            end
        end

        if Div > 0
            @inbounds for j in 1:Div
                div_quant = divergences[j]
                grids[out_idx].grid[i] = div_quant
                out_idx += 1
            end
        end

        if C > 0
            @inbounds for j in 1:C
                curl_quant = curls[j]
                @inbounds for d in 1:3
                    grids[out_idx].grid[i] = curl_quant[d]
                    out_idx += 1
                end
            end
        end
        i += stride
    end
    return nothing
end

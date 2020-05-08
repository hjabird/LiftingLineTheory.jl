#
# VortexFilament.jl
#
#   Operations on vortex filaments. 
#
#   These isn't actually a vortex filament object. Instead, they are 
#   represented by two matrices and an array:
#       - f_starts:     Matrix of size N by 3
#       - f_ends:       Matrix of size N by 3
#       - f_strengths:  Vector of size N  
#   This is easy to use with CVortex.jl. An empty VortexFilament type 
#   is used to make sure the correct function is called and so anyone
#   reading your code ends up knowing what you were doing with your poorly
#   named matrices / vector.
#
# Copyright HJA Bird 2019 - 2020
#
#==============================================================================#

struct VortexFilament # We just want to be able to use the type
end

function to_vortex_particles(
    ::Type{VortexFilament},
    fil_start::Vector{<:Real},
    fil_end::Vector{<:Real},
    fil_str::Real,
    particle_separation::Float64
    ) :: Tuple{Matrix{Float32}, Matrix{Float32}}
    
    @assert(length(fil_start)==3)
    @assert(length(fil_end)==3)
    @assert(isfinite(fil_str))
    @assert(particle_separation > 0)

    dir = fil_end - fil_start
    len = sqrt(dir[1]^2 + dir[2]^2 + dir[3]^2)
    np = Int64(ceil(len  / particle_separation))
    lpstrs = repeat(dir'; inner=(np, 1)) * fil_str / np
    lppos = zeros(np, 3)
    for j = 1:np
        lppos[j,:] = fil_start
        lppos[j,:] += (j - 0.5) * dir / np
    end
    return lppos, lpstrs
end

function to_vortex_particles(
    ::Type{VortexFilament},
    fil_start::Matrix{<:Real},
    fil_end::Matrix{<:Real},
    fil_str::Vector{<:Real},
    particle_separation::Float64
    ) :: Tuple{Matrix{Float32}, Matrix{Float32}}
    @assert(size(fil_start)==size(fil_end))
    @assert(size(fil_start)[2]==3)
    @assert(length(fil_str)==size(fil_start)[1])
    @assert(particle_separation > 0)

    nfils = length(fil_str)

    particle_strs = zeros(0,3)
    particle_locs = zeros(0,3)
    for i = 1:nfils
        lppos, lpstrs = to_vortex_particles(
            VortexFilament,
            fil_end[i,:], fil_start[i,:], fil_str[i],
            particle_separation)
        particle_strs = cat(particle_strs, lpstrs; dims=1)
        particle_locs = cat(particle_locs, lppos; dims=1)
    end
    @assert(all(isfinite.(particle_locs)))
    @assert(all(isfinite.(particle_strs)))
    return particle_locs, particle_strs
end

function to_vtk(::Type{VortexFilament},
    fil_starts::Matrix{<:Real}, fil_ends::Matrix{<:Real}, 
    fil_strs::Vector{<:Real}, filename::String;
    translation::Vector{<:Real}=[0,0,0]) :: Nothing

    @assert(size(fil_starts)[2] == 3)
    @assert(size(fil_ends)[2] == 3)
    @assert(size(fil_strs)[1] == size(fil_starts)[1])
    @assert(size(fil_strs)[1] == size(fil_ends)[1])

    if prod(size(fil_strs))!= 0
        nfils = size(fil_starts)[1]
        points = vcat(fil_starts, fil_ends)
        points .+= translation'
        cells = Vector{WriteVTK.MeshCell}(undef, nfils)        
        celldata = fil_strs
        for i = 1 : nfils
            cells[i] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LINE, 
                [i, i+nfils])
        end
    else
        points = zeros(0, 3)
        cells = Vector{WriteVTK.MeshCell}(undef, 0)
        cell_str = zeros(0)
    end
    vtkfile = WriteVTK.vtk_grid(filename, points', cells)
    WriteVTK.vtk_cell_data(vtkfile, celldata, "Vorticity_density")
    WriteVTK.vtk_save(vtkfile)
    return
end

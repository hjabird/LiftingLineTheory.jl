#
# ParticleGroup2D.jl
#
# Represent a group of vortex particles in a way easy to use with CVortex.jl
#
# Copyright HJA Bird 2019-2020
#
################################################################################


mutable struct ParticleGroup2D
    positions :: Matrix{Float32} # An N by 3 Matrix
    vorts :: Vector{Float32}

    function ParticleGroup2D()
        return new(zeros(Float32, 0, 2), zeros(Float32, 0))
    end

end

function check(a::ParticleGroup2D; do_assert::Bool=false) :: Bool
    finite_pos = all(isfinite.(a.positions))
    finite_vort = all(isfinite.(a.vorts))
    correct_pos_dim = size(a.positions)[2]==2
    matching_len = size(a.positions)[1] == length(a.vorts)
    if do_assert
        @assert(finite_pos, "Non-finite positions.")
        @assert(finite_vort, "Non-finite vorticities")
        @assert(correct_pos_dim, "Positions should be "*
            "(N, 2) matrix, but is "*string(size(a.positions))*".")
        @assert(matching_len, "Number of particles suggested by "*
            "vorticity vector doesn't match that suggested by "*
            "positions matrix. Vorticity suggests "*string(length(a.vorts))*
            " particles and positions suggests "*
            string(size(a.positions)[1])*".")
    end
    return finite_pos && finite_vort && correct_pos_dim && matching_len
end

function total_vorticity(a::ParticleGroup2D)
    return sum(a.vorts)
end

function to_vtk(
    a::ParticleGroup2D,
    filename::String;
    translation::Vector{<:Real}=[0,0]) :: Nothing

    particle_locs = a.positions
    particle_vorts = a.vorts
    
    @assert(size(particle_locs)[2] == 2)
    @assert(size(particle_locs)[1] == length(particle_vorts))

    nvorts = size(particle_locs)[1]
    if nvorts != 0
        points = particle_locs
        points .+= translation'
        cells = Vector{WriteVTK.MeshCell}(undef, nvorts)        
        celldata = particle_vorts
        for i = 1 : nvorts
            cells[i] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_VERTEX, [i])
        end
    else
        points = zeros(0, 2)
        cells = Vector{WriteVTK.MeshCell}(undef, 0)
        celldata = zeros(0)
    end
    vtkfile = WriteVTK.vtk_grid(filename, points', cells)
    if nvorts != 0
        WriteVTK.vtk_cell_data(vtkfile, celldata', "Vorticity")
    end
    WriteVTK.vtk_save(vtkfile)
    return
end


#
# ParticleGroup3D.jl
#
# Represent a group of vortex particles in a way easy to use with CVortex.jl
#
# Copyright HJA Bird 2020
#
################################################################################


mutable struct ParticleGroup3D
    positions :: Matrix{Float32} # An N by 3 Matrix
    vorts :: Matrix{Float32} # An N by 3 Matrix

    function ParticleGroup3D()
        return new(zeros(Float32, 0, 3), zeros(Float32, 0, 3))
    end

end

function check(a::ParticleGroup3D; do_assert::Bool=false) :: Bool
    by3_pos = size(a.positions)[2] == 3
    by3_vort = size(a.vorts)[2] == 3
    same_len = size(a.positions)[1] == size(a.vorts)[1]
    finite_geom = all(isfinite.(a.positions))
    finite_vort = all(isfinite.(a.vorts))
    if do_assert
        @assert(by3_pos, "The positions array is not (N, 3) but "*
            "is "*string(size(a.positions))*".")
        @assert(by3_vort, "The positions array is not (N, 3) but "*
            "is "*string(size(a.vorts))*".")
        @assert(same_len, "Positions and vorts matrices are for different "*
            "number of particles. Positions suggests "*
            string(size(a.positions)[1])*" particles and vorts suggests "*
            string(size(a.positions)[1])*" particles.")
        @assert(finite_geom, "Geometry contains non-finite values.")
        @assert(finite_vort, "Vorticity contains non-finite values.")
    end
    return finite_geom && finite_vort && same_len && by3_pos && by3_vort
end

function num_particles(a::ParticleGroup3D) :: Int64
    return size(a.positions)[1]
end

function add_particles!(a::ParticleGroup3D, 
    positions::Matrix{Float32}, vorts::Matrix{Float32}) :: Nothing

    @assert(size(positions)[2]==3, "Positions should be (N,3) but instead "*
        "has dimensions "*size(positions)*".")
    @assert(size(vorts)[2]==3, "Vorticities should be (N,3) but instead "*
        "has dimensions "*size(vorts)*".")
    @assert(size(positions)[1]==size(vorts)[1], "The number of particles"*
        " suggested by the size of the positions and vorticities "*
        "matrix is different. Positions suggests "*string(size(positions)[2])*
        " particles and vorticities suggests "*string(size(vorts)[2])*
        " particles.")
    if size(a.positions)[2]!=3 || size(a.vorts)[2]!=3
        check(a; do_assert=true)
    end
    
    a.positions = cat(a.positions, positions; dims=1)
    a.vorts = cat(a.vorts, vorts; dims=1)
    return
end

function to_vtk(
    a::ParticleGroup3D,
    filename::String;
    translation::Vector{<:Real}=[0,0,0])

    particle_locs = a.positions
    particle_vorts = a.vorts
    
    @assert(size(particle_locs)[2] == 3)
    @assert(size(particle_vorts)[2] == 3)
    @assert(size(particle_vorts) == size(particle_locs))

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
        points = zeros(0, 3)
        cells = Vector{WriteVTK.MeshCell}(undef, 0)
        celldata = zeros(0, 3)
    end
    vtkfile = WriteVTK.vtk_grid(filename, points', cells)
    if nvorts != 0
        WriteVTK.vtk_cell_data(vtkfile, celldata', "Vorticity")
    end
    WriteVTK.vtk_save(vtkfile)
    return
end

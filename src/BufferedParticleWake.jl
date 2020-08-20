#
# BufferedParticleWake.jl
#
# A set of vortex particles with a vortex lattice buffer region for attachment
# to time varying vortex lattices.
#
# Copyright HJA Bird 2020
#
################################################################################

mutable struct BufferedParticleWake

    lattice_buffer :: VortexLattice # We add and remove columns from this.
    edge_fil_strs :: Vector{Float32}    # We need a little trickery 
                                        # lattice-particle interface.
    wake_particles :: ParticleGroup3D   # Vortex particles.

    function BufferedParticleWake(;
        lattice::VortexLattice=VortexLattice(zeros(1,1,3)),
        edge_fil_strs::Vector{<:Real}=zeros(Float32,0),
        wake_particles::ParticleGroup3D=ParticleGroup3D()
        )
        new(lattice, edge_fil_strs, wake_particles)
    end
end

function BufferedParticleWake(
    initial_geom_row::Matrix{<:Real}
    ) :: BufferedParticleWake

    ni, nj = size(initial_geom_row)
    @assert(nj == 3)
    @assert(ni > 1)
    edge_fil_strs = zeros(Float32, ni-1)
    lattice = VortexLattice(reshape(deepcopy(initial_geom_row), :, 1, 3))
    return BufferedParticleWake(;
        lattice=lattice, edge_fil_strs=edge_fil_strs)
end

function get_vertices(
    a::BufferedParticleWake;
    include_tether=true) :: Matrix{Float64}
    # Changes to ordering here must be reflected in the
    # set_vertices! method.

    if include_tether
        fil_verts = get_vertices(a.lattice_buffer)
    else
        ni, nj = size(a.lattice_buffer.vertices)
        fil_verts = reshape(a.lattice_buffer.vertices[:, 1:nj-1, :], :, 3)
    end
    p_verts = a.wake_particles.positions
    return cat(fil_verts, p_verts; dims=1)
end

function set_vertices!(
    a::BufferedParticleWake, 
    vertices::Matrix{<:Real};
    include_tether=true
    ) :: Nothing

    # Changes to ordering here must be reflected in the
    # get_vertices method.
    
    @assert(size(vertices)[2] == 3, "The input vertices Matrix "*
        "is expected to be (N, 3), but actually has size "*
        string(size(vertices))*".")
    @assert(all(isfinite.(vertices)),
        "Not all input vertices are finite.")

    nv_in = size(vertices)[1]
    if include_tether
        lattice_verts = prod(size(a.lattice_buffer.vertices)[1:2])
    else
        lni, lnj = size(a.lattice_buffer.vertices)
        lattice_verts = lni * (lnj-1)
    end
    particle_verts = size(a.wake_particles.positions)[1]
    n_buffer_verts = lattice_verts + particle_verts
    @assert(nv_in == n_buffer_verts, "Number of input vertices does not "*
        "match number of BufferedParticleWake vertices. There are "*
            string(nv_in) * " input vertices for an "*
            string(n_buffer_verts) * " vertex wake.")

    if include_tether
        set_vertices!(a.lattice_buffer, vertices[1:lattice_verts,:])
        a.wake_particles.positions = vertices[lattice_verts+1:end, :]
    else
        reshp_verts = reshape(vertices[1:lattice_verts, :], lni, lnj-1, 3)
        a.lattice_buffer.vertices[:, 1:lnj-1, :] = reshp_verts
        a.wake_particles.positions = vertices[lattice_verts+1:end, :]
    end

    check(a.lattice_buffer; do_assert=true)
    check(a.wake_particles; do_assert=true)
    return
end

function check(a::BufferedParticleWake; do_assert::Bool=false) :: Bool
    buffer_good = check(a.lattice_buffer; do_assert=do_assert)
    particles_good = check(a.wake_particles; do_assert=do_assert)
    finite_edge_strs = all(isfinite.(a.edge_fil_strs))
    finite_edge_dim = (length(a.edge_fil_strs) == 
        size(a.lattice_buffer.strengths)[1])
    if do_assert
        @assert(finite_edge_strs, 
            "Non-finite edge filament strength corrections.")
        @assert(finite_edge_dim, 
            "Edge filament strength correction vector does not match "*
            "the vortex lattice dimensions. Length is "*
            string(length(a.edge_fil_strs))*" but it should be "*
            string(size(a.lattice_buffer.strengths)[1])*".")
    end
    return buffer_good && particles_good && finite_edge_strs && finite_edge_dim
end

#   Transfer some of the vortex lattice buffer to vortex particles.
#   
#   The vortex lattice buffer is built up by adding rows as 
#   a simulation continues. This is manually transferred to 
#   particles with a correction for the filament strenghts
#   on the edge of the vortex lattice.
function buffer_to_particles(
    a::BufferedParticleWake,
    particle_separation::Real;
    buffer_rows::Int64=1) :: Nothing

    @assert(buffer_rows >= 1,
        "There must be 1 or more wake buffer rows! Tried to use "*
        string(buffer_rows)*" rows.")
    @assert(particle_separation > 0,
        "Particle separation must be +ve. Given "*
        string(particle_separation)*".")
    check(a; do_assert=true)

    # Step 1: Get the filaments we're converting.
    ei, ej = size(a.lattice_buffer.strengths)
    if ej-buffer_rows > 0   # Do we have anything we can convert?
        to_p_lattice = extract_sublattice(a.lattice_buffer,
            1, ei, 1, ej-buffer_rows)
        cf_starts, cf_ends, cf_str = centre_filaments(to_p_lattice)
        ef_starts, ef_ends, ef_str = edge_filaments(
            to_p_lattice; i_min=true, i_max=true)
        jmin_starts, jmin_ends, jmin_strs = edge_filaments(
            to_p_lattice; j_min=true)
        # Correct for the edge fil strs.
        jmin_strs -= a.edge_fil_strs
        # Put 'em all together for convection.
        f_starts = cat(cf_starts, ef_starts, jmin_starts; dims=1)
        f_ends = cat(cf_ends, ef_ends, jmin_ends; dims=1)
        f_strs = cat(cf_str, ef_str, jmin_strs; dims=1)

        # Step 2: Convert some filaments to particles.
        p_locs, p_vorts = to_vortex_particles(
            VortexFilament, f_starts, f_ends, f_strs,
            particle_separation)
        mask = reshape(sum(abs.(p_vorts); dims=2) .!= 0,:)
        add_particles!(a.wake_particles, p_locs[mask,:], p_vorts[mask,:])

        # Step 3: Remove converted filaments from buffer.
        new_lattice = extract_sublattice(a.lattice_buffer,
        1, ei, ej-buffer_rows+1, ej)
        ~, ~, jmax_strs = edge_filaments(to_p_lattice; j_max=true)
        a.lattice_buffer = new_lattice
        a.edge_fil_strs = -jmax_strs

        # Step 4: Profit.
    end
    return
end

#   Obtain the vortex filaments from the wake.
function get_filaments(a::BufferedParticleWake;
    exclude_buffer_column::Bool=false
    ) :: Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}}
    # This is a largely a copy and past from the VortexLattice method
    # of to_filaments, except that the rstr function is modified
    # to take account of the a.edge_fil_strs object and also
    # can ignore the buffer that we might be varying.

    check(a.lattice_buffer; do_assert=true)
    @assert(length(a.edge_fil_strs)==size(a.lattice_buffer.strengths)[1],
        "Edge filament correction vector in BufferParticleWake is "*
        "incorrect length. Length is "*string(length(a.edge_fil_strs))*
        " but it should match a edge of length "*
        string(size(a.lattice_buffer.strengths)[1])*".")

    if !exclude_buffer_column
        r_verts = a.lattice_buffer.vertices
        r_strs = a.lattice_buffer.strengths
    else
        tmp_i, tmp_j = size(a.lattice_buffer.strengths)
        if tmp_j > 1
            truncated_lattice = extract_sublattice(
                a.lattice_buffer, 1, tmp_i, 1, tmp_j-1)
            r_verts = truncated_lattice.vertices
            r_strs = truncated_lattice.strengths
        else
            return zeros(0,3), zeros(0,3), zeros(0)
        end
    end

    nif = (size(r_verts)[1]-1) * (size(r_verts)[2]) # i-dir (streamwise)
    njf = (size(r_verts)[1]) * (size(r_verts)[2]-1) # j-dir (spanwise)
    nf = nif + njf
    fstarts = zeros(nf, 3)
    fends = zeros(nf, 3)
    fstrs = zeros(nf)
    acc = 1
    function rstr(i, j)
        if (i > 0) && (j > 0) && (i <= size(r_strs)[1]) && (j <= size(r_strs)[2])
            ret = r_strs[i, j]
        #elseif (j == size(r_strs)[2]+1) && (i>0) && (i <= size(r_strs)[1])
        #   ret = a.edge_fil_strs[i]
        elseif (j == 0) && (i>0) && (i <= size(r_strs)[1])
           ret = a.edge_fil_strs[i]
        else
            ret = 0
        end
        return ret
    end
    # (i --- i+1)
    for i = 1 : size(r_verts)[1]-1
        for j = 1 : size(r_verts)[2]
            fstarts[acc, :] = r_verts[i, j, :]
            fends[acc, :] = r_verts[i+1, j, :]
            fstrs[acc] = rstr(i, j) - rstr(i, j-1)
            acc += 1
        end
    end
    @assert(acc - 1 == nif)
    # (j --- j+1)
    for i = 1 : size(r_verts)[1]
        for j = 1 : size(r_verts)[2]-1
            fstarts[acc, :] = r_verts[i, j, :]
            fends[acc, :] = r_verts[i, j+1, :]
            fstrs[acc] = rstr(i-1, j) - rstr(i, j)
            acc += 1
        end
    end
    @assert(acc - 1 == nif + njf)
    return fstarts, fends, fstrs
end

#   Obtaiin the particles from the wake.
function get_vortex_particles(a::BufferedParticleWake
    ) :: Tuple{Matrix{Float32}, Matrix{Float32}}
    return a.wake_particles.positions, a.wake_particles.vorts
end


function add_new_buffer_row!(
    a::BufferedParticleWake,
    geometry::Matrix{<:Real};
    ring_strengths::Vector{<:Real}=zeros(0)
    ) :: Nothing

    vi, vj = size(a.lattice_buffer.vertices)
    ri = size(a.lattice_buffer.strengths)[1]
    @assert(size(geometry)[2]==3)
    @assert(size(geometry)[1]==vi)
    if length(ring_strengths) == 0
        ring_strengths = zeros(ri)
    else
        @assert(length(ring_strengths) == ri,
            "Length of ring strength vector should be equal to existing "*
            "lattice size, but it is "*string(length(ring_strengths))*
            " when "*string(ri)*" is needed.")
    end

    j_idx = add_column!(a.lattice_buffer)
    a.lattice_buffer.vertices[:,j_idx+1,:] = geometry
    return
end


function create_initial_rings!(a::BufferedParticleWake,
    initial_displacements::Matrix{<:Real}) :: Nothing

    @assert(size(initial_displacements)[2]==3)
    @assert(size(a.lattice_buffer.vertices)[2]==1)
    a.lattice_buffer.vertices = cat(
        a.lattice_buffer.vertices .+ reshape(
            initial_displacements, :, 1, 3),
        a.lattice_buffer.vertices
        ; dims=2)
    a.lattice_buffer.strengths = zeros(size(a.lattice_buffer.vertices)[1]-1, 1)
    return
end

# Get the influence of the most recently inputted row of the
# buffer wake.
function ring_influence_matrix(
    a::BufferedParticleWake, 
    mespoints::Matrix{Float64}, 
    mesnormals::Matrix{Float64}
    ) :: Matrix{Float64}
    
    @assert(size(mespoints)[2] == 3)
    @assert(size(mespoints) == size(mesnormals))

    ni, nj = size(a.lattice_buffer.strengths)
    most_recent_col = extract_sublattice(
        a.lattice_buffer, 1, ni, nj, nj)
    vel_inf, ~ = ring_influence_matrix(
        most_recent_col, mespoints, mesnormals)
    @assert(size(vel_inf)==(size(mespoints)[1], ni))
    return vel_inf
end


function redistribute_particles!(
    a::BufferedParticleWake,
    particle_spacing::Float64;
    redistrubtion_function::CVortex.RedistributionFunction=m4p_redistribution(),
	negligible_vort::Real=0.1,
	max_new_particles::Integer=-1
    ) :: Nothing

    check(a.wake_particles; do_assert=true)
    
    plocs = a.wake_particles.positions
    pvorts = a.wake_particles.vorts
    nplocs, npvorts = redistribute_particles_on_grid(
        plocs, pvorts, 
        redistrubtion_function, 
        particle_spacing;
        negligible_vort=negligible_vort,
        max_new_particles=max_new_particles)
    a.wake_particles.positions = nplocs
    a.wake_particles.vorts = npvorts
    return
end


function to_vtk(a::BufferedParticleWake,
    filename::String; 
    translation::Vector{<:Real}=[0., 0., 0.])

    check(a; do_assert=true)

    nparticles = num_particles(a.wake_particles)
    fs, fe, fstr = get_filaments(a)
    nfils = size(fs)[1]

    if nfils != 0
        fpoints = vcat(fs, fe)
        fpoints .+= translation'
        fcells = Vector{WriteVTK.MeshCell}(undef, nfils)        
        fcelldata = fstr
        for i = 1 : nfils
            fcells[i] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LINE, 
                [i, i+nfils])
        end
    else
        fpoints = zeros(0, 3)
        fcells = Vector{WriteVTK.MeshCell}(undef, 0)
        fcell_str = zeros(0)
    end

    if nparticles != 0
        ppoints = a.wake_particles.positions
        ppoints .+= translation'
        pcells = Vector{WriteVTK.MeshCell}(undef, nparticles)        
        pcelldata = a.wake_particles.vorts
        for i = 1 : nparticles
            pcells[i] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_VERTEX, [i+nfils*2])
        end
    else
        ppoints = zeros(0, 3)
        pcells = Vector{WriteVTK.MeshCell}(undef, 0)
        pcelldata = zeros(0, 3)
    end
    vtkfile = WriteVTK.vtk_grid(filename, 
        cat(fpoints', ppoints'; dims=2), 
        cat(fcells, pcells; dims=1))

    WriteVTK.vtk_cell_data(vtkfile, 
        cat(ones(nfils,3)' .* NaN, pcelldata'; dims=2), "Vorticity")
    WriteVTK.vtk_cell_data(vtkfile, 
        cat(fcelldata, ones(nparticles) .* NaN; dims=1), "Vorticity_density")

    WriteVTK.vtk_save(vtkfile)
    return
end


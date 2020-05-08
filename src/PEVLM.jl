
mutable struct PEVLM
    free_stream_vel :: Float64
    kinematics :: RigidKinematics2D
    reference_wing_lattice :: VortexLattice # TE at jmax.

    wing_lattice :: VortexLattice       
    old_wing_lattice :: VortexLattice

    te_wake :: BufferedParticleWake

    regularisation_radius :: Float64
    regularisation_kernel :: CVortex.RegularisationFunction
    dt :: Float64
    current_time :: Float64

    initialised :: Bool

    # Detailed constructor
    function PEVLM(;
        ref_wing_lattice :: VortexLattice,
        kinematics :: RigidKinematics2D,
        dt :: Float64 = -1.,
        free_stream_vel :: Real = 1.,
        current_time :: Float64 = 0.,
        regularisation_radius :: Float64 = -1.,
        regularisation_kernel :: CVortex.RegularisationFunction = gaussian_regularisation())

        szw = size(ref_wing_lattice.vertices)
        wing_lattice = VortexLattice(zeros(szw))
        old_wing_lattice = VortexLattice(zeros(szw))

        new(free_stream_vel, kinematics,
            ref_wing_lattice,
            wing_lattice, old_wing_lattice, 
            BufferedParticleWake(),
            regularisation_radius, regularisation_kernel,
            dt, current_time, false)
    end
end

function PEVLM(
    wing :: StraightAnalyticWing,
    kinematics :: RigidKinematics2D;
    dt :: Float64 = -1.,
    free_stream_vel :: Real = 1.,
    current_time :: Float64 = 0.,
    wing_discretisation_chordwise::Vector{<:Real}=collect(-1:0.1:1),
    wing_discretisation_spanwise::Vector{<:Real}=collect(-1:0.1:1),
    regularisation_radius :: Float64 = -1.,
    regularisation_kernel :: CVortex.RegularisationFunction = gaussian_regularisation()
    ) :: PEVLM

    function geom_fn(x,y)
        # x and y are swapped because of the desired indexing scheme.
        xp = y * wing.chord_fn(y * wing.semispan)/2
        yp = x * wing.semispan
        return [xp, yp, 0]
    end
    ref_lattice = VortexLattice(geom_fn;
        ys=wing_discretisation_chordwise, 
        xs=wing_discretisation_spanwise)
    if dt == -1
        dt = 0.05 * wing.chord_fn(0) / free_stream_vel
    end
    @assert(dt != 0.0)
    if regularisation_radius == -1.
        regularisation_radius = dt * free_stream_vel * 1.5
    end
    
    return PEVLM(
        ref_wing_lattice=ref_lattice,
        kinematics=kinematics,
        dt=dt,
        free_stream_vel=free_stream_vel,
        current_time=current_time,
        regularisation_kernel=regularisation_kernel,
        regularisation_radius=regularisation_radius
    )
end

function advance_one_step(a::PEVLM) :: Nothing
    if !a.initialised
        update_wing_lattice!(a)
        initialise_wake!(a)
        a.initialised = true
    end

    a.current_time += a.dt
    convect_wake!(a)
    a.old_wing_lattice = a.wing_lattice
    update_wing_lattice!(a)
    update_wake_lattice!(a)
    compute_wing_vortex_strengths!(a)
    return
end


function convect_wake!(a::PEVLM) :: Nothing
    check(a.wing_lattice; do_assert=true)
    check(a.te_wake; do_assert=true)

    # Get vorticity sources.
    fwi_starts, fwi_ends, fwi_str = to_filaments(a.wing_lattice)
    fwa_starts, fwa_ends, fwa_str = get_filaments(a.te_wake)
    f_starts = cat(fwi_starts, fwa_starts; dims=1)
    f_ends = cat(fwi_ends, fwa_ends; dims=1)
    f_str = cat(fwi_str, fwa_str; dims=1)
    p_locs, p_vorts = get_vortex_particles(a.te_wake)
    @assert(all(isfinite.(p_vorts)))

    # Get near wake lattice vel.
    te_points = get_vertices(a.te_wake)
    te_vels = filament_induced_velocity(f_starts, f_ends, f_str, te_points)
    @assert(all(isfinite.(te_vels)))
    te_vels[:,1] .+= a.free_stream_vel
    @assert(all(isfinite.(te_vels)))
    te_vels += CVortex.particle_induced_velocity(
        p_locs, p_vorts, te_points,
        a.regularisation_kernel, a.regularisation_radius)
    @assert(all(isfinite.(te_vels)))

    # Get particle wake vels + dvorts.
    p_dvorts = filament_induced_dvort(
        f_starts, f_ends, f_str, p_locs, p_vorts )
    @assert(all(isfinite.(p_dvorts)), "A non-finite rate of change "*
        "of vorticity was induced on the wake particles.")
    p_dvorts += CVortex.particle_induced_dvort(
        p_locs, p_vorts, 
        p_locs, p_vorts,
        a.regularisation_kernel, a.regularisation_radius)
    @assert(all(isfinite.(p_dvorts)), "A non-finite rate of change "*
        "of vorticity was induced on the wake particles.")

    dt = a.dt
    # Set new near wake geometry
    new_te_points = te_points + te_vels .* dt
    set_vertices!(a.te_wake, new_te_points)
    # Set new particle wake vorts.
    a.te_wake.wake_particles.vorts += p_dvorts .* dt
    return
end


function update_wing_lattice!(a::PEVLM) :: Nothing
    z_offset = z_pos(a.kinematics, a.current_time)
    aoa = AoA(a.kinematics, a.current_time)
    piv_x = pivot_position(a.kinematics)
    
    new_geometry = get_vertices(a.reference_wing_lattice)
    dx = new_geometry[:,1] .- piv_x
    new_geometry[:,1] = cos(aoa) * dx .+ piv_x
    new_geometry[:,3] = sin(aoa) * dx
    new_geometry[:,3] .+= z_offset
    set_vertices!(a.wing_lattice, new_geometry)
    return
end


function update_wake_lattice!(a::PEVLM; buffer_rows::Int=1) :: Nothing

    wake_verts = reshape(a.wing_lattice.vertices[:,end,:],:,3)
    add_new_buffer_row!(a.te_wake, wake_verts)
    buffer_to_particles(a.te_wake,
        2 * a.regularisation_radius / 2.0;  # Particle separation.
        buffer_rows=buffer_rows)
    return
end 


function compute_wing_vortex_strengths!(a::PEVLM) :: Nothing

    @assert(size(a.wing_lattice.vertices)[1]==
        size(a.te_wake.lattice_buffer.vertices)[1],
        "Mismatch in wing and wake discretisations. Wing was "*
        string(size(a.wing_lattice.vertices)[1])*" vortex rings wide and the"*
        " was "*string(size(a.te_wake.lattice_buffer.vertices)[1])*" wide.")

    centres = ring_centres(a.wing_lattice)
    normals = ring_normals(a.wing_lattice)
    centres = reshape(centres, size(centres)[1]*size(centres)[2], 3)
    normals = reshape(normals, size(normals)[1]*size(normals)[2], 3)

    # The wing's self influence
    inf_mat, wing_ring_idxs = 
        ring_influence_matrix(a.wing_lattice, centres, normals)
    # We need 1 row of the near_wake to impose the K-J condition.
    near_wake_inf_mat = ring_influence_matrix(a.te_wake, centres, normals)
    # Now add to near_wake influence matrix. 
    inf_mat[:, wing_ring_idxs[:, size(a.wing_lattice.strengths)[2]]] += near_wake_inf_mat

    # Get const influences from the near_wake and free stream.
    ext_vel = external_induced_vel(a, centres)

    # Get the velocity of the points on the wing.
    z_vel = dzdt(a.kinematics, a.current_time) 
    z_off = z_pos(a.kinematics, a.current_time)
    daoadt = dAoAdt(a.kinematics, a.current_time)
    piv_x = pivot_position(a.kinematics)
    wing_vel = zeros(size(centres))
    wing_vel[:,3] .= z_vel
    wing_vel[:,1] += daoadt .* (centres[:,3] .- z_off)
    wing_vel[:,2] += daoadt .* (centres[:,1] .- piv_x)

    # Get the normal velocity from the wing / ext vels.
    ext_inf = sum((ext_vel+wing_vel) .* normals, dims=2) # Dot product of each row.#

    # Yay, finally something we can solve
    ring_strengths = inf_mat \ ext_inf

    rs = map(i->ring_strengths[i], wing_ring_idxs)
    a.wing_lattice.strengths = rs
    a.te_wake.lattice_buffer.strengths[:,end] = 
        rs[wing_ring_idxs[:, size(a.wing_lattice.strengths)[2]]]
    if !all(isfinite.(rs))
        @warn("Non-finite wing vorticities computed.")
    end
    return
end


function initialise_wake!(a::PEVLM) :: Nothing
    vertices = reshape(a.reference_wing_lattice.vertices[:,end:end,:], :, 3)
    a.te_wake = BufferedParticleWake(vertices)
    return
end


function external_induced_vel(
    a::PEVLM,
    points :: Matrix{Float64}
    ) :: Matrix{Float64}

    @assert(size(points)[2] == 3)
    @assert(all(isfinite.(points)), "Non-finite input point.")
    check(a.te_wake; do_assert=true)

    # Get const influences free stream
    ext_vel = zeros(size(points))
    ext_vel[:,1] .= a.free_stream_vel
    # Influence of near wake
    fstarts, fends, fstrs = get_filaments(
        a.te_wake; exclude_buffer_column=true )
    near_wake_ind_vel = CVortex.filament_induced_velocity(
        fstarts, fends, fstrs, points )
    ext_vel += near_wake_ind_vel
    # Influence of particle wake.
    p_locs, p_vorts = get_vortex_particles(a.te_wake)
    ext_vel += CVortex.particle_induced_velocity(
        p_locs, p_vorts, points,
        a.regularisation_kernel, a.regularisation_radius)
    @assert(all(isfinite.(ext_vel)), 
        "Non-finite value in external induced vels.")
    return ext_vel
end


#= IO FUNCTIONS =============================================================#

function to_vtk(a::PEVLM,
    filename_start::String, file_no::Int;
    translation::Vector{<:Real}=[0,0,0]) :: Nothing

    wingfile = filename_start * "_wing_" * string(file_no)
    te_wakefile = filename_start * "_wake_" * string(file_no)
    to_vtk(a.wing_lattice, wingfile; translation=translation)
    to_vtk(a.te_wake, te_wakefile; translation=translation)
    return
end

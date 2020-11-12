#
# UnsteadyVortexLatticeMethod.jl
#
# Unsteady Vortex Lattice Method
#
# Copyright HJAB 2020
#
################################################################################

mutable struct UnsteadyVortexLatticeMethod
    free_stream_vel :: Float64
    kinematics :: RigidKinematics2D
    reference_wing_lattice :: VortexLattice # TE at jmax.

    wing_lattice :: VortexLattice       
    old_wing_lattice :: VortexLattice
    wake :: VortexLattice               # The col at jmax is most recently shed.

    dt :: Float64
    current_time :: Float64

    initialised :: Bool

    # Detailed constructor
    function UnsteadyVortexLatticeMethod(;
        ref_wing_lattice :: VortexLattice,
        kinematics :: RigidKinematics2D,
        dt :: Float64 = -1.,
        free_stream_vel :: Real = 1.,
        current_time :: Float64 = 0.)

        szw = size(ref_wing_lattice.vertices)
        wing_lattice = VortexLattice(zeros(szw))
        old_wing_lattice = VortexLattice(zeros(szw))
        wake = VortexLattice(zeros(szw[1],1,3))

        new(free_stream_vel, kinematics,
            ref_wing_lattice,
            wing_lattice, old_wing_lattice, wake,
            dt, current_time, false)
    end
end

function UnsteadyVortexLatticeMethod(
    wing :: StraightAnalyticWing,
    kinematics :: RigidKinematics2D;
    dt :: Float64 = -1.,
    free_stream_vel :: Real = 1.,
    current_time :: Float64 = 0.,
    wing_discretisation_chordwise::Vector{<:Real}=-cos.(range(0, pi; length=10)),
    wing_discretisation_spanwise::Vector{<:Real}=cos.(range(0, pi; length=10))
    ) :: UnsteadyVortexLatticeMethod

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
    
    return UnsteadyVortexLatticeMethod(
        ref_wing_lattice=ref_lattice,
        kinematics=kinematics,
        dt=dt,
        free_stream_vel=free_stream_vel,
        current_time=current_time
    )
end


function advance_one_step(a::UnsteadyVortexLatticeMethod) :: Nothing
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


function convect_wake!(a::UnsteadyVortexLatticeMethod) :: Nothing
    # Convert vortex lattices in the problem to CVortex filaments
    fwi_starts, fwi_ends, fwi_str = to_filaments(a.wing_lattice)
    fwa_starts, fwa_ends, fwa_str = to_filaments(a.wake)
    f_starts = cat(fwi_starts, fwa_starts; dims=1)
    f_ends = cat(fwi_ends, fwa_ends; dims=1)
    f_str = cat(fwi_str, fwa_str; dims=1)
    points = get_vertices(a.wake)
    vels = filament_induced_velocity(f_starts, f_ends, f_str, points)
    vels[:,1] .+= a.free_stream_vel
    dt = a.dt
    new_points = points + vels .* dt
    set_vertices!(a.wake, new_points)
    return
end


function update_wing_lattice!(a::UnsteadyVortexLatticeMethod) :: Nothing
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


function update_wake_lattice!(a::UnsteadyVortexLatticeMethod) :: Nothing
    # We need to add a new row to the lattice and reconnect it to the wing.
    j_idx = add_column!(a.wake)
    ni = extent_i(a.wake)
    @assert(ni == extent_i(a.wing_lattice))
    a.wake.vertices[:,j_idx+1,:] = a.wing_lattice.vertices[:,end,:]
    a.wake.strengths[:,j_idx] = -a.wing_lattice.strengths[:,end]
    return
end 


function compute_wing_vortex_strengths!(a::UnsteadyVortexLatticeMethod) :: Nothing
    centres = ring_centres(a.wing_lattice)
    normals = ring_normals(a.wing_lattice)
    centres = reshape(centres, size(centres)[1]*size(centres)[2], 3)
    normals = reshape(normals, size(normals)[1]*size(normals)[2], 3)

    # The wing's self influence
    @time inf_mat, wing_ring_idxs = 
        ring_influence_matrix(a.wing_lattice, centres, normals)

    # Get const influences from the wake and free stream.
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
    ext_inf = sum((-wing_vel-ext_vel) .* normals, dims=2) # Dot product of each row.#

    # Yay, finally something we can solve
    ring_strengths = inf_mat \ ext_inf

    rs = map(i->ring_strengths[i], wing_ring_idxs)
    a.wing_lattice.strengths = rs
    return
end


function initialise_wake!(a::UnsteadyVortexLatticeMethod) :: Nothing
    ni, nj = size(a.wing_lattice.strengths);
    verts = a.wing_lattice.vertices[:,nj:nj,:]
    a.wake = VortexLattice(verts)
    return
end


function external_induced_vel(
    a::UnsteadyVortexLatticeMethod,
    points :: Matrix{Float64}
    ) :: Matrix{Float64}

    @assert(size(points)[2]==3)

    # Get const influences from the wake and free stream.
    ext_vel = zeros(size(points))
    ext_vel[:,1] .= a.free_stream_vel
    nwi, nwj = size(a.wake.strengths)
    fstarts, fends, fstrs = to_filaments(a.wake)
    wake_ind_vel = CVortex.filament_induced_velocity(
        fstarts, fends, fstrs, points )
    ext_vel += wake_ind_vel
    return ext_vel
end

function pressure_distribution(a::UnsteadyVortexLatticeMethod; density::Float64=1.) :: Matrix{Float64}
    # See page 427 of Katz and Plotkin
    # Rate of change of ring vorticities.
    cvd = a.wing_lattice.strengths
    ovd = a.old_wing_lattice.strengths
    dvdt = (cvd - ovd) ./ a.dt
    # Geometry
    rcs = ring_centres(a.wing_lattice)
    tangentd1, tangentd2 = normalised_ring_tangents(a.wing_lattice)
    ring_widthd1, ring_widthd2 = ring_widths(a.wing_lattice)
    # Velocity at ring centres.
    ni, nj, ~ = size(rcs)
    rc_vec = zeros(ni*nj, 3)
    surf_vels = zeros(ni, nj, 3)
    for i = 1 : ni
        for j = 1 : nj
            rc_vec[(i-1)*nj+j,:] = rcs[i,j,:]
        end
    end
    # Get const influences from the near_wake and free stream.
    ext_vel = nonwing_ind_vel(a, rc_vec)
    # Get the velocity of the points on the wing.
    z_vel = dzdt(a.kinematics, a.current_time) 
    z_off = z_pos(a.kinematics, a.current_time)
    daoadt = dAoAdt(a.kinematics, a.current_time)
    piv_x = pivot_position(a.kinematics)
    wing_vel = zeros(size(rc_vec))
    wing_vel[:,3] .= z_vel
    wing_vel[:,1] += daoadt .* (rc_vec[:,3] .- z_off)
    wing_vel[:,2] += daoadt .* (rc_vec[:,1] .- piv_x)
    ext_vel = ext_vel-wing_vel
    for i = 1 : ni
        for j = 1 : nj
            surf_vels[i,j,:] = ext_vel[(i-1)*nj+j,:]
        end
    end
    # Vorticity deriv wrt/ grid.
    vd1, vd2 = vorticity_derivatives(a.wing_lattice) # include c_ij etc terms. 
    
    # Compute pressure differences
    pres = zeros(ni,nj)
    for i = 1 : ni
        for j = 1 : nj
            pres[i,j] = (sum(surf_vels[i,j,:].*tangentd1[i,j,:]) * vd1[i,j] +
                sum(surf_vels[i,j,:].*tangentd2[i,j,:]) * vd2[i,j] +
                dvdt[i,j])
        end
    end
    pres = pres .* density
    return pres
end

function to_vtk(a::UnsteadyVortexLatticeMethod,
    filename_start::String, file_no::Int;
    translation::Vector{<:Real}=[0,0,0]) :: Nothing

    wingfile = filename_start * "_wing_" * string(file_no)
    wakefile = filename_start * "_wake_" * string(file_no)
    to_vtk(a.wing_lattice, wingfile; translation=translation)
    to_vtk(a.wake, wakefile; translation=translation)
    return
end

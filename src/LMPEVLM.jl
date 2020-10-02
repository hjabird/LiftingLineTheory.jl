#
# LMPEVLM.jl
#
# L(ESP) Modulated Particle Enhanced Vortex Lattice Method
#
# Copyright HJAB 2020
#
################################################################################

mutable struct LMPEVLM
    free_stream_vel :: Float64
    kinematics :: RigidKinematics2D
    reference_wing_lattice :: VortexLattice # TE at jmax.

    wing_lattice :: VortexLattice       
    old_wing_lattice :: VortexLattice

    te_wake :: BufferedParticleWake # trailing edge wake
    le_wake :: BufferedParticleWake # Leading edge wake
    lt_wake :: BufferedParticleWake # Left tip wake
    rt_wake :: BufferedParticleWake # Right tip wake

    critical_le_vorticity :: Vector{Float64}
	kinematic_viscocity :: Float64
    regularisation_radius :: Float64
    regularisation_kernel :: CVortex.RegularisationFunction
    dt :: Float64
    current_time :: Float64

    initialised :: Bool
    wing_self_influence_matrix :: Matrix{Float32}
    wing_self_influence_matrix_ring_mapping :: Matrix{Int64}

    # Detailed constructor
    function LMPEVLM(;
        ref_wing_lattice :: VortexLattice,
        kinematics :: RigidKinematics2D,
        dt :: Float64 = -1.,
        free_stream_vel :: Real = 1.,
        current_time :: Float64 = 0.,
        critical_le_vorticity :: Vector{Float64} = zeros(0),
		kinematic_viscocity :: Float64 = 0.,
        regularisation_radius :: Float64 = -1.,
        regularisation_kernel :: CVortex.RegularisationFunction = gaussian_regularisation()
        ) :: LMPEVLM

        szw = size(ref_wing_lattice.vertices)
        wing_lattice = VortexLattice(zeros(szw))
        old_wing_lattice = VortexLattice(zeros(szw))

        new(free_stream_vel, kinematics,
            ref_wing_lattice,
            wing_lattice, old_wing_lattice, 
            BufferedParticleWake(), BufferedParticleWake(),
            BufferedParticleWake(), BufferedParticleWake(),
            critical_le_vorticity, kinematic_viscocity,
            regularisation_radius, regularisation_kernel,
            dt, current_time, false, zeros(Float32,0,0), zeros(Int64,0,0))
    end
end

function LMPEVLM(
    wing :: StraightAnalyticWing,
    kinematics :: RigidKinematics2D;
    dt :: Float64 = -1.,
    free_stream_vel :: Real = 1.,
    current_time :: Float64 = 0.,
    wing_discretisation_chordwise::Vector{<:Real}=collect(range(-1, 1; length=20)), 
    wing_discretisation_spanwise::Vector{<:Real}=cos.(range(0, pi; length=40)),
    a0_critical :: Float64 = -1.0,
	kinematic_viscocity :: Float64 = 0.0,
    regularisation_radius :: Float64 = -1.0,
    regularisation_kernel :: CVortex.RegularisationFunction = gaussian_regularisation()
    ) :: LMPEVLM

    ## GEOMETRY
    function geom_fn(x,y)
        # x and y are swapped because of the desired indexing scheme.
        xp = y * wing.chord_fn(x * wing.semispan) / 2
        yp = x * wing.semispan
        return [xp, yp, 0]
    end
    # We want to put the ring geometry 1/4 of a ring back.
    wdc = deepcopy(wing_discretisation_chordwise)
    cwdc = deepcopy(wdc)
    cwdc[1:end-1] = wdc[1:end-1] + 0.25 * (wdc[2:end]-wdc[1:end-1])
    cwdc[end] = wdc[end] + 0.25 * (wdc[end] - wdc[end-1])
    ref_lattice = VortexLattice(geom_fn;
        ys=cwdc, 
        xs=wing_discretisation_spanwise)
    if dt == -1
        dt = 0.05 * wing.chord_fn(0) / free_stream_vel
    end
    
    @assert(dt != 0.0)
    if regularisation_radius == -1.
        regularisation_radius = dt * free_stream_vel 
    end

    if a0_critical == -1
        critical_le_vorticity = fill(9e99, extent_i(ref_lattice))
    else
        dx1 = ref_lattice.vertices[:,2,1] - ref_lattice.vertices[:,1,1]
        dx1 = (dx1[1:end-1] + dx1[2:end]) / 2
        yps = ref_lattice.vertices[:,1,2]
        yps = (yps[1:end-1] + yps[2:end]) / 2
        semichords = map(y->wing.chord_fn(y)/2, yps)
        corr = free_stream_vel * 2 * semichords .* (
            acos.(1 .- dx1 ./ semichords) +
            sin.(acos.(1 .- dx1 ./ semichords)))
        critical_le_vorticity = corr * a0_critical
    end
    @assert(all(critical_le_vorticity .>= 0))

    return LMPEVLM(
        ref_wing_lattice=ref_lattice,
        kinematics=kinematics,
        dt=dt,
        free_stream_vel=free_stream_vel,
        current_time=current_time,
        critical_le_vorticity=critical_le_vorticity,
		kinematic_viscocity=kinematic_viscocity,
        regularisation_kernel=regularisation_kernel,
        regularisation_radius=regularisation_radius
    )
end

function advance_one_step(a::LMPEVLM) :: Nothing
    if !a.initialised
        update_wing_lattice!(a)
        initialise_wake!(a)
        a.initialised = true
    end

    a.current_time += a.dt
    convect_wake!(a)
    a.old_wing_lattice = a.wing_lattice
    update_wing_lattice!(a)
    update_wake_lattice!(a; buffer_rows=5)
    compute_wing_vortex_strengths!(a)
    if num_particles(a.te_wake.wake_particles) > 500000
        @error("Probably blown up!")
    end
    return
end


function convect_wake!(a::LMPEVLM) :: Nothing
    check(a.wing_lattice; do_assert=true)
    check(a.te_wake; do_assert=true)
    check(a.le_wake; do_assert=true)

    nte_vorts = num_particles(a.te_wake.wake_particles)
    nle_vorts = num_particles(a.le_wake.wake_particles)
    nlt_vorts = num_particles(a.lt_wake.wake_particles)
    nrt_vorts = num_particles(a.rt_wake.wake_particles)
    nte_verts = num_vertices(a.te_wake)
    nle_verts = num_vertices(a.le_wake)
    nlt_verts = num_vertices(a.lt_wake)
    nrt_verts = num_vertices(a.rt_wake)
    function get_dpoints()
        return vcat(
            get_vertices(a.te_wake), get_vertices(a.le_wake),
            get_vertices(a.lt_wake), get_vertices(a.rt_wake))
    end
    function get_dvorts()
        ~, p_vorts_te = get_vortex_particles(a.te_wake)
        ~, p_vorts_le = get_vortex_particles(a.le_wake)
        ~, p_vorts_lt = get_vortex_particles(a.lt_wake)
        ~, p_vorts_rt = get_vortex_particles(a.rt_wake)
        vorts = vcat(p_vorts_te, p_vorts_le, p_vorts_lt, p_vorts_rt)
        return vorts
    end
    function set_dpoints(new_points)
        set_vertices!(a.te_wake, new_points[1:nte_verts,:])
        set_vertices!(a.le_wake, new_points[
            nte_verts+1:nte_verts+nle_verts,:])
        set_vertices!(a.lt_wake, new_points[
            nte_verts+nle_verts+1:nte_verts+nle_verts+nlt_verts,:])
        set_vertices!(a.rt_wake, new_points[
            nte_verts+nle_verts+nlt_verts+1:end,:])
        return
    end
    function set_dvorts(new_vorts)
        a.te_wake.wake_particles.vorts = new_vorts[1:nte_vorts,:]
        a.le_wake.wake_particles.vorts = new_vorts[
            nte_vorts+1:nte_vorts+nle_vorts,:]
        a.lt_wake.wake_particles.vorts = new_vorts[
            nte_vorts+nle_vorts+1:nte_vorts+nle_vorts+nlt_vorts,:]
        a.rt_wake.wake_particles.vorts = new_vorts[
            nte_vorts+nle_vorts+nlt_vorts+1:end,:]
        return
    end
    function vel_method()
        return field_velocity(a, 
            vcat(
            get_vertices(a.te_wake), get_vertices(a.le_wake),
            get_vertices(a.lt_wake), get_vertices(a.rt_wake)))
    end
    function dvort_method() 
        p_locs_te, p_vorts_te = get_vortex_particles(a.te_wake)
        p_locs_le, p_vorts_le = get_vortex_particles(a.le_wake)
        p_locs_lt, p_vorts_lt = get_vortex_particles(a.lt_wake)
        p_locs_rt, p_vorts_rt = get_vortex_particles(a.rt_wake)
        p_vorts = vcat(p_vorts_te, p_vorts_le, p_vorts_lt, p_vorts_rt)
        p_locs = vcat(p_locs_te, p_locs_le, p_locs_lt, p_locs_rt)
        p_locse, p_vortse = everything_as_particles(a)
        p_dvorts = CVortex.particle_induced_dvort(
            p_locse, p_vortse, 
            p_locs, p_vorts,
            a.regularisation_kernel, a.regularisation_radius)  
        if a.kinematic_viscocity != 0
            vol = (a.regularisation_radius * (1/1.5))^3 # Should be consistent w/ remeshing
            CVortex.particle_visc_induced_dvort(p_locs, p_vorts,
                vol, p_locs, p_vorts,
                vol, a.regularisation_kernel, a.regularisation_radius,
                a.kinematic_viscocity)
        end
        return p_dvorts
    end
    ode = PointsVortsODE(
        get_dpoints, get_dvorts, vel_method, dvort_method,
        set_dpoints, set_dvorts )
    runge_kutta_4_step(ode, a.dt)
    return
end


function update_wing_lattice!(a::LMPEVLM) :: Nothing
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


function update_wake_lattice!(
    a::LMPEVLM; 
    buffer_rows::Int=1) :: Nothing
    update_te_wake_lattice!(a; buffer_rows=buffer_rows)
    update_le_wake_lattice!(a; buffer_rows=buffer_rows)
    update_tip_wake_lattices!(a; buffer_rows=buffer_rows)
    return
end 


function update_te_wake_lattice!(a::LMPEVLM; buffer_rows::Int=1) :: Nothing

    te_wake_verts = reshape(a.wing_lattice.vertices[:,end,:],:,3)
    add_new_buffer_row!(a.te_wake, te_wake_verts)
    a.te_wake.lattice_buffer.strengths[:, end] = -a.wing_lattice.strengths[:, end]
    buffer_to_particles(a.te_wake,
        2 * a.regularisation_radius / 2.0;  # Particle separation.
        buffer_rows=buffer_rows)
    return
end 


function update_le_wake_lattice!(a::LMPEVLM; buffer_rows::Int=1) :: Nothing

    le_pos = wing_le_position(LMPEVLM, a.wing_lattice)
    le_filament_verts = reshape(a.wing_lattice.vertices[:,1,:],:,3)
    last_shed_pos = a.le_wake.lattice_buffer.vertices[:,end-2,:]
    newpos = le_pos + (last_shed_pos - le_pos) ./ 3

    add_new_buffer_row!(a.le_wake, le_filament_verts)
    a.le_wake.lattice_buffer.vertices[:, end - 2,:] = newpos
    a.le_wake.lattice_buffer.vertices[:, end - 1,:] = le_pos
    a.le_wake.lattice_buffer.vertices[:, end,    :] = le_filament_verts
    # The strengths[:,end] and [:,end-1] should have been equal before new column.
    # So we need to zero the new [:,end-1] and [:,end]
    a.le_wake.lattice_buffer.strengths[:, end-1] .= 0.
    a.le_wake.lattice_buffer.strengths[:, end] .= 0.

    #a.le_wake.lattice_buffer.strengths[:, end] = -le_wake_strs
    buffer_to_particles(a.le_wake,
        2 * a.regularisation_radius / 2.0;  # Particle separation.
        buffer_rows=buffer_rows+2)
    return
end 


function update_tip_wake_lattices!(a::LMPEVLM; buffer_rows::Int=1) :: Nothing
    # On one tip we have to do -ve strengths
    lt_wake_verts = reshape(a.wing_lattice.vertices[1,:,:],:,3)
    add_new_buffer_row!(a.lt_wake, lt_wake_verts)
    a.lt_wake.lattice_buffer.strengths[:, end] = -a.wing_lattice.strengths[1, :]
    buffer_to_particles(a.lt_wake,
        2 * a.regularisation_radius / 2.0;  # Particle separation.
        buffer_rows=buffer_rows)

    rt_wake_verts = reshape(a.wing_lattice.vertices[end,:,:],:,3)
    add_new_buffer_row!(a.rt_wake, rt_wake_verts)
    a.rt_wake.lattice_buffer.strengths[:, end] = a.wing_lattice.strengths[end, :]
    buffer_to_particles(a.rt_wake,
        2 * a.regularisation_radius / 2.0;  # Particle separation.
        buffer_rows=buffer_rows)
    return
end 


function initialise_wake!(a::LMPEVLM) :: Nothing
    # The trailing edge wake
    ni, nj = size(a.wing_lattice.strengths);
    verts_te = a.wing_lattice.vertices[:,nj:nj,:]
    a.te_wake.lattice_buffer = VortexLattice(verts_te)
    a.te_wake.edge_fil_strs = zeros(ni)
    check(a.wing_lattice; do_assert=true)

    # The leading edge wake
    verts_fle = a.wing_lattice.vertices[:,1:1,:]
    verts_le = wing_le_position(LMPEVLM, a.wing_lattice)
    vel = -wing_surface_velocity(a, verts_le)
    vel[:, 1] .+= a.free_stream_vel
    free_edge = verts_le + a.dt * vel * 1.5
    lattice_verts = cat(
        reshape(free_edge, ni+1, 1, 3),
        reshape(verts_le, ni+1, 1, 3),
        verts_fle; 
        dims=2)
    a.le_wake.lattice_buffer = VortexLattice(lattice_verts)
    a.le_wake.edge_fil_strs = zeros(ni)

    # The wing tips
    verts_flt = a.wing_lattice.vertices[1:1,:,:]
    vel = -wing_surface_velocity(a, verts_flt[1,:,:])
    vel[:, 1] .+= a.free_stream_vel
    free_edge = verts_flt[1,:,:] + a.dt * vel * 1.5
    lattice_verts = cat(
        reshape(free_edge, nj+1, 1, 3),
        reshape(verts_flt, nj+1, 1, 3); 
        dims=2)
    a.lt_wake.lattice_buffer = VortexLattice(lattice_verts)
    a.lt_wake.edge_fil_strs = zeros(nj)

    verts_frt = a.wing_lattice.vertices[end:end,:,:]
    vel = -wing_surface_velocity(a, verts_frt[1,:,:])
    vel[:, 1] .+= a.free_stream_vel
    free_edge = verts_frt[1,:,:] + a.dt * vel * 1.5
    lattice_verts = cat(
        reshape(free_edge, nj+1, 1, 3),
        reshape(verts_frt, nj+1, 1, 3); 
        dims=2)
    a.rt_wake.lattice_buffer = VortexLattice(lattice_verts)
    a.rt_wake.edge_fil_strs = zeros(nj)

    check(a.wing_lattice; do_assert=true)
    return
end


function wing_surface_velocity(
    a::LMPEVLM,
    points_on_wing_surface :: Matrix{<:Real}) :: Matrix{<:Real}

    pnts = points_on_wing_surface
    @assert(size(pnts)[2] == 3)
    z_vel = dzdt(a.kinematics, a.current_time) 
    z_off = z_pos(a.kinematics, a.current_time)
    daoadt = dAoAdt(a.kinematics, a.current_time)
    piv_x = pivot_position(a.kinematics)
    wing_vel = zeros(size(pnts))
    wing_vel[:,3] .= z_vel
    wing_vel[:,1] += daoadt .* (pnts[:,3] .- z_off)
    wing_vel[:,2] += daoadt .* (pnts[:,1] .- piv_x)
    return wing_vel
end


function wing_centres_and_normals(
    a::LMPEVLM
    ) :: Tuple{Matrix{Float64}, Matrix{Float64}}

    centres = ring_centres(a.wing_lattice)
    normals = ring_normals(a.wing_lattice)
    centres = reshape(centres, size(centres)[1]*size(centres)[2], 3)
    normals = reshape(normals, size(normals)[1]*size(normals)[2], 3)
    return centres, normals
end


function wing_self_influence_matrix(a::LMPEVLM)
    winfmat = a.wing_self_influence_matrix
    ringmap = a.wing_self_influence_matrix_ring_mapping
    if (length(winfmat) == 0) || (length(ringmap) == 0)
        centres, normals = wing_centres_and_normals(a)
        winfmat, ringmap = 
            ring_influence_matrix(a.wing_lattice, centres, normals)
        a.wing_self_influence_matrix = winfmat
        a.wing_self_influence_matrix_ring_mapping = ringmap
    end
    return winfmat, ringmap
end


function compute_wing_vortex_strengths!(a::LMPEVLM
    ) :: Nothing

    @assert(size(a.wing_lattice.vertices)[1]==
        size(a.te_wake.lattice_buffer.vertices)[1],
        "Mismatch in wing and wake discretisations. Wing was "*
        string(size(a.wing_lattice.vertices)[1])*" vortex rings wide and the"*
        " was "*string(size(a.te_wake.lattice_buffer.vertices)[1])*" wide.")
    check(a.wing_lattice; do_assert=true)
    check(a.le_wake; do_assert=true)

    # Shedding LE mask determines the point on the LE where we we impose 
    # the LESP criterion.
    ni_w, nj_w = size(a.wing_lattice.strengths)
    ni_le, nj_le = size(a.le_wake.lattice_buffer.strengths)
    @assert(ni_le == ni_w)
    shedding_le_mask = fill(false, ni_w)
    centres, normals = wing_centres_and_normals(a)
    w_inf_mat, w_ring_idxs = wing_self_influence_matrix(a)
    wing_le_idxs = w_ring_idxs[:,1]
    # The influence matrix for the LEV - we'll selectively use bits.
    lev_sublattice = extract_sublattice(
        a.le_wake.lattice_buffer, 1, ni_w, nj_le-1, nj_le)
    le_inf_matx, le_ring_idxs = 
        ring_influence_matrix(lev_sublattice, centres, normals)
    # The two rows have to share the same ring strength. 
    le_inf_mat = mapreduce(
        i->le_inf_matx[:,le_ring_idxs[i,1]]+le_inf_matx[:,le_ring_idxs[i,2]],
        vcat, [1:ni_le])

    # We don't need to do anything with the wing tips unless we want
    # to do a critical A0 kind of thing.
    
    # Get const influences from the near_wake and free stream.
    ext_vel = nonwing_ind_vel(a, centres)
    wing_vel = wing_surface_velocity(a, centres)

    # Iteration is required for the lesp criterion.
    critical_vorts = a.critical_le_vorticity
    leading_edge_vorticity = zeros(ni_w)
    le_vort_sgn = zeros(ni_le)
    ring_strengths = zeros(size(centres)[1])
    while true
        le_vort_sgn = map(x-> x > 0. ? 1. : -1., leading_edge_vorticity)
        # Construct the modification to the influence matrix.
        inf_mat = deepcopy(w_inf_mat)
        inf_mat[:, wing_le_idxs[shedding_le_mask]] += le_inf_mat[:, shedding_le_mask]

        # Construct the modification to the ext vel vector.
        lesp_const_vel = zeros(size(centres)[1])
        lesp_const_vel += le_inf_mat[:,shedding_le_mask] * (
            -le_vort_sgn[shedding_le_mask] .* critical_vorts[shedding_le_mask])
        # Dot product of each row.
        ext_inf = sum((wing_vel-ext_vel) .* normals, dims=2) - lesp_const_vel

        # Solve for ring strengths.
        ring_strengths = inf_mat \ ext_inf

        # Do we need to modify the shedding locations and try again?
        leading_edge_vorticity = map(i->ring_strengths[i], w_ring_idxs[:,1])
        o_le_shedding_mask = abs.(leading_edge_vorticity) .> critical_vorts
        if o_le_shedding_mask == shedding_le_mask
            break
        else
            shedding_le_mask = o_le_shedding_mask
        end
    end

    rs = map(i->ring_strengths[i], w_ring_idxs)
    a.wing_lattice.strengths = rs
    a.le_wake.lattice_buffer.strengths[shedding_le_mask, end] = 
        (rs[shedding_le_mask,1] - 
        le_vort_sgn[shedding_le_mask] .* critical_vorts[shedding_le_mask])
    a.le_wake.lattice_buffer.strengths[shedding_le_mask, end-1] = 
        a.le_wake.lattice_buffer.strengths[shedding_le_mask, end] 
    
    return
end


function nonwing_ind_vel(
    a::LMPEVLM,
    points :: Matrix{Float64}
    ) :: Matrix{Float64}

    @assert(size(points)[2] == 3)
    @assert(all(isfinite.(points)), "Non-finite input point.")
    check(a.te_wake; do_assert=true)

    # Get const influences free stream
    ext_vel = zeros(size(points))
    ext_vel[:,1] .= a.free_stream_vel
    # Influence of near wake
    ftew_starts, ftew_ends, ftew_strs = get_filaments( a.te_wake )
    # 1st two rows of LE should be zeroed at this point so we don't have
    # to extract them.
    flew_starts, flew_ends, flew_strs = get_filaments( a.le_wake )
    fltw_starts, fltw_ends, fltw_strs = get_filaments( a.lt_wake )
    frtw_starts, frtw_ends, frtw_strs = get_filaments( a.rt_wake )
    fstarts = cat(ftew_starts, flew_starts, fltw_starts, frtw_starts; dims=1)
    fends = cat(ftew_ends, flew_ends, fltw_ends, frtw_ends; dims=1)
    fstrs = cat(ftew_strs, flew_strs, fltw_strs, frtw_strs; dims=1)
    near_wake_ind_vel = CVortex.filament_induced_velocity(
        fstarts, fends, fstrs, points )
    ext_vel += near_wake_ind_vel
    # Influence of particle wake.
    p_tlocs, p_tvorts = get_vortex_particles(a.te_wake)
    p_llocs, p_lvorts = get_vortex_particles(a.le_wake)
    p_ltlocs, p_ltvorts = get_vortex_particles(a.lt_wake)
    p_rtlocs, p_rtvorts = get_vortex_particles(a.rt_wake)
    ext_vel += CVortex.particle_induced_velocity(
        cat(p_tlocs, p_llocs, p_ltlocs, p_rtlocs; dims=1), 
        cat(p_tvorts, p_lvorts, p_ltvorts, p_rtvorts; dims=1), points,
        CVortex.singular_regularisation(), a.regularisation_radius)
    @assert(all(isfinite.(ext_vel)), 
        "Non-finite value in external induced vels.")
    return ext_vel
end


function everything_as_particles(
    a::LMPEVLM) :: Tuple{Matrix{Float32}, Matrix{Float32}}

    # Get vorticity sources.
    fwi_starts, fwi_ends, fwi_str = to_filaments(a.wing_lattice)
    fwa_starts, fwa_ends, fwa_str = get_filaments(a.te_wake)
    flewa_starts, flewa_ends, flewa_str = get_filaments(a.le_wake)
    fltwa_starts, fltwa_ends, fltwa_str = get_filaments(a.lt_wake)
    frtwa_starts, frtwa_ends, frtwa_str = get_filaments(a.rt_wake)
    f_starts = cat(fwi_starts, flewa_starts, fltwa_starts, 
        frtwa_starts, fwa_starts; dims=1)
    f_ends = cat(fwi_ends, flewa_ends, fltwa_ends, 
        frtwa_ends, fwa_ends; dims=1)
    f_str = cat(fwi_str, flewa_str, fltwa_str, 
        frtwa_str, fwa_str; dims=1)
    # Turn the filaments into particles so that our problem
    # doesn't explode.
    p_locs_f, p_vorts_f = to_vortex_particles( VortexFilament,
        f_starts, f_ends, f_str, a.regularisation_radius )
    p_locs_te, p_vorts_te = get_vortex_particles(a.te_wake)
    p_locs_le, p_vorts_le = get_vortex_particles(a.le_wake)
    p_locs_lt, p_vorts_lt = get_vortex_particles(a.lt_wake)
    p_locs_rt, p_vorts_rt = get_vortex_particles(a.rt_wake)
    p_locs = cat(p_locs_te, p_locs_le, p_locs_lt, 
        p_locs_rt, p_locs_f; dims=1)
    p_vorts = cat(p_vorts_te, p_vorts_le, p_vorts_lt, 
        p_vorts_rt, p_vorts_f; dims=1)
    return p_locs, p_vorts
end


function field_velocity(
    a::LMPEVLM,
    points :: Matrix{Float64}
    ) :: Matrix{Float32}
    @assert(size(points)[2] == 3)
    @assert(all(isfinite.(points)), "Non-finite input point.")

    p_locs, p_vorts = everything_as_particles(a)
    @assert(all(isfinite.(p_vorts)))

    vels = CVortex.particle_induced_velocity(
        p_locs, p_vorts, points,
        a.regularisation_kernel, a.regularisation_radius)
    @assert(all(isfinite.(vels)))
    vels[:,1] .+= a.free_stream_vel
    return vels
end


function field_induced_dvort(
    a::LMPEVLM,
    particle_locs :: Matrix{Float32},
    particle_vorts:: Matrix{Float32}
    ) :: Matrix{Float32}
    @assert(size(particle_locs)[2] == 3)
    @assert(all(isfinite.(particle_locs)), "Non-finite input point.")
    @assert(size(particle_vorts) == size(particle_locs))
    @assert(all(isfinite.(particle_vorts)), "Non-finite input point.")

    p_locs, p_vorts = everything_as_particles(a)
    @assert(all(isfinite.(p_vorts)))

    dvorts = CVortex.particle_induced_dvort(
        p_locs, p_vorts, particle_locs, particle_vorts,
        a.regularisation_kernel, a.regularisation_radius)
    @assert(all(isfinite.(dvorts)))
    return dvorts
end


function field_vorticity(
    a::LMPEVLM,
    mes_locs::Matrix{Float32}) :: Matrix{Float32}

    p_locs, p_vorts = everything_as_particles(a)
    @assert(all(isfinite.(p_vorts)))

    vorts = CVortex.particle_field_vorticity(
        p_locs, p_vorts, mes_locs,
        a.regularisation_kernel, a.regularisation_radius)
    @assert(all(isfinite.(vorts)))
    return vorts
end


function wing_le_position(
    ::Type{LMPEVLM},
    foil_lattice :: VortexLattice) :: Matrix{Float64}

    posns = foil_lattice.vertices
    p0 = posns[:,1,:]
    p1 = posns[:,2,:]
    ple = p0 - 0.25 * (p1 - p0)
    return ple
end


function transfer_all_particles_to_te_wake!(a::LMPEVLM)
    tepos, tevort = get_vortex_particles(a.te_wake)
    lepos, levort = get_vortex_particles(a.le_wake)
    ltpos, ltvort = get_vortex_particles(a.lt_wake)
    rtpos, rtvort = get_vortex_particles(a.rt_wake)
    pos = vcat(tepos, lepos, ltpos, rtpos)
    vort = vcat(tevort, levort, ltvort, rtvort)
    a.te_wake.wake_particles.positions=pos 
    a.te_wake.wake_particles.vorts=vort
    a.le_wake.wake_particles.positions=zeros(0,3)
    a.le_wake.wake_particles.vorts=zeros(0,3)
    a.lt_wake.wake_particles.positions=zeros(0,3)
    a.lt_wake.wake_particles.vorts=zeros(0,3)
    a.rt_wake.wake_particles.positions=zeros(0,3)
    a.rt_wake.wake_particles.vorts=zeros(0,3)
    return
end

function redistribute_wake!(a::LMPEVLM)
    transfer_all_particles_to_te_wake!(a)
    p_locs, p_vorts = get_vortex_particles(a.te_wake)
    nppos, npvort, ~ = CVortex.redistribute_particles_on_grid(
        p_locs, p_vorts, 
        CVortex.lambda3_redistribution(),
        a.regularisation_radius * (1/1.5); # Needs to be consistent w/ dvort
        negligible_vort=0.25)
    a.te_wake.wake_particles.positions = nppos;
    a.te_wake.wake_particles.vorts = npvort
    return;
end

function relax_wake!(a::LMPEVLM; 
    relaxation_parameter::Float64=0.3) :: Nothing
    transfer_all_particles_to_te_wake!(a)
    p_locs, p_vorts = get_vortex_particles(a.te_wake)
    npvort= CVortex.particle_pedrizzetti_relaxation(
        p_locs, p_vorts, relaxation_parameter,
        a.regularisation_kernel, a.regularisation_radius )
    a.te_wake.wake_particles.vorts = npvort
    return;
end


#= IO FUNCTIONS =============================================================#

function to_vtk(a::LMPEVLM,
    filename_start::String, file_no::Int;
    translation::Vector{<:Real}=[0,0,0]) :: Nothing

    wingfile = filename_start * "_wing_" * string(file_no)
    te_wakefile = filename_start * "_tewake_" * string(file_no)
    le_wakefile = filename_start * "_lewake_" * string(file_no)
    lt_wakefile = filename_start * "_ltwake_" * string(file_no)
    rt_wakefile = filename_start * "_rtwake_" * string(file_no)
    to_vtk(a.wing_lattice, wingfile; translation=translation)
    to_vtk(a.te_wake, te_wakefile; translation=translation)
    to_vtk(a.le_wake, le_wakefile; translation=translation)
    to_vtk(a.lt_wake, lt_wakefile; translation=translation)
    to_vtk(a.rt_wake, rt_wakefile; translation=translation)
    return
end

function cross_section_vels(a::LMPEVLM, filename::String, y::Float64)
    xs = collect(-1:0.025:1)
    zs = collect(-1:0.025:1)
    nx = length(xs)
    nz = length(zs)
    coord_msh = zeros(nx, nz, 3)
    for ix = 1 : nx
        for iz = 1 : nz
            coord_msh[ix,iz,:] = [xs[ix], y, zs[iz]]
        end
    end
    coord_msh = reshape(coord_msh,:,3)
    vels = field_velocity(a, coord_msh)
    idxs = collect(1 : size(vels)[1])
    idxs = reshape(idxs, nx, nz)

    points = coord_msh
    ncells = (nx-1)*(nz-1)
    cells = Vector{WriteVTK.MeshCell}(undef, ncells)
    acc = 0
    for ix = 1 : nx-1
        for iz = 1 : nz-1
            acc += 1
            cells[acc] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUAD, 
                [idxs[ix,iz], idxs[ix,iz+1], idxs[ix+1,iz+1], idxs[ix+1,iz]])
        end
    end

    vtkfile = WriteVTK.vtk_grid(filename, points', cells)
    WriteVTK.vtk_point_data(vtkfile, vels', "Velocity")
    WriteVTK.vtk_save(vtkfile)
end

function export_vorticity_field(
    a::LMPEVLM, 
    filename::String;
    resolution::Float64=-1.0,
    bounds=[]) :: Nothing

    if resolution == -1.0
        resolution = a.regularisation_radius / 3
    end
    if length(bounds) == 0
        pts = a.te_wake.wake_particles.positions
        pts = vcat(pts, get_vertices(a.wing_lattice))
        pts = vcat(pts, a.le_wake.wake_particles.positions)
        pts = vcat(pts, a.lt_wake.wake_particles.positions)
        pts = vcat(pts, a.rt_wake.wake_particles.positions)
        bounds = [
            minimum(pts[:,1]), maximum(pts[:,1]),
            minimum(pts[:,2]), maximum(pts[:,2]),
            minimum(pts[:,3]), maximum(pts[:,3])]
    end
    @assert(resolution > 0)
    @assert(length(bounds) == 6)
    @assert(all(isfinite.(bounds)))
    res = resolution

    xs = collect(bounds[1]-res:res:bounds[2]+res)
    ys = collect(bounds[3]-res:res:bounds[4]+res)
    zs = collect(bounds[5]-res:res:bounds[6]+res)
    println(bounds)
    @assert(length(xs)>1)
    @assert(length(ys)>1)
    @assert(length(zs)>1)
    nx = length(xs)
    ny = length(ys)
    nz = length(zs)
    coord_msh = zeros(Float32, nx, ny, nz, 3)
    for ix = 1 : nx
        for iy = 1 : ny
            for iz = 1 : nz
                coord_msh[ix,iy,iz,:] = [xs[ix], ys[iy], zs[iz]]
            end
        end
    end
    coord_msh = reshape(coord_msh,:,3)
    vorts = field_vorticity(a, coord_msh)
    idxs = collect(1 : size(vorts)[1])
    idxs = reshape(idxs, nx, ny, nz)

    points = coord_msh
    ncells = (nx-1)*(nz-1)*(ny-1)
    cells = Vector{WriteVTK.MeshCell}(undef, ncells)
    acc = 0
    for ix = 1 : nx-1
        for iy = 1 : ny-1
            for iz = 1 : nz-1
                acc += 1
                cells[acc] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_HEXAHEDRON, 
                    [idxs[ix,iy,iz], idxs[ix+1,iy,iz], idxs[ix+1,iy+1,iz], idxs[ix,iy+1,iz],
                    idxs[ix,iy,iz+1], idxs[ix+1,iy,iz+1], idxs[ix+1,iy+1,iz+1], idxs[ix,iy+1,iz+1]])
            end
        end
    end

    vtkfile = WriteVTK.vtk_grid(filename, points', cells)
    WriteVTK.vtk_point_data(vtkfile, vorts', "Vorticity")
    WriteVTK.vtk_save(vtkfile)
    return
end

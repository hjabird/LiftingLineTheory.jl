#
# PEVLM.jl
#
# Particle Enhanced Vortex Lattice Method
#
# Copyright HJAB 2020
#
################################################################################

mutable struct PEVLM
    free_stream_vel :: Float64
    kinematics :: RigidKinematics2D

    reference_wing_geometry :: StraightAnalyticWing
    wing_discretisation_chordwise :: Vector{Float64}
    wing_discretisation_spanwise :: Vector{Float64}

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
        reference_wing_geometry :: StraightAnalyticWing,
        kinematics :: RigidKinematics2D,
        wing_discretisation_chordwise :: Union{Vector{<:Real}, Int} = zeros(0),
        wing_discretisation_spanwise :: Union{Vector{<:Real}, Int} = zeros(0),
        dt :: Float64 = -1.,
        free_stream_vel :: Real = 1.,
        current_time :: Float64 = 0.,
        regularisation_radius :: Float64 = -1.,
        regularisation_kernel :: CVortex.RegularisationFunction = gaussian_regularisation())

        if dt == -1
            dt = 1/20 * reference_wing_geometry.chord_fn(0) / free_stream_vel
        end
        @assert(dt != 0.0)
        if regularisation_radius == -1.
            regularisation_radius = dt * free_stream_vel * 1.5
        end

        ## GEOMETRY
        if length(wing_discretisation_chordwise) == 0
            wing_discretisation_chordwise = -cos.(range(0, pi; length=15))
        elseif typeof(wing_discretisation_chordwise) <: Int
            @assert(wing_discretisation_chordwise > 1)
            wing_discretisation_chordwise = -cos.(range(0, pi; 
            length=wing_discretisation_chordwise))
        end
        if length(wing_discretisation_spanwise) == 0|| typeof(wing_discretisation_chordwise) <: Int
            ar = aspect_ratio(reference_wing_geometry)
            num_spanswise = Int64(round(10 + ar * 3))
            wing_discretisation_spanwise = cos.(range(0, pi; length=num_spanswise))
        elseif typeof(wing_discretisation_spanwise) <: Int
            @assert(wing_discretisation_spanwise > 1)
            wing_discretisation_spanwise = -cos.(range(0, pi; 
                length=wing_discretisation_spanwise))
        end
        function geom_fn(x,y)
            # x and y are swapped because of the desired indexing scheme.
            xp = y * reference_wing_geometry.chord_fn(
                x * reference_wing_geometry.semispan) / 2
            yp = x * reference_wing_geometry.semispan
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

        szw = size(ref_lattice.vertices)
        wing_lattice = VortexLattice(zeros(szw))
        old_wing_lattice = VortexLattice(zeros(szw))

        new(free_stream_vel, kinematics,
            reference_wing_geometry, wing_discretisation_chordwise,
            wing_discretisation_spanwise,
            ref_lattice, wing_lattice, old_wing_lattice, 
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
    wing_discretisation_chordwise::Union{Vector{<:Real}, Int}=zeros(0),
    wing_discretisation_spanwise::Union{Vector{<:Real}, Int}=zeros(0),
    regularisation_radius :: Float64 = -1.,
    regularisation_kernel :: CVortex.RegularisationFunction = gaussian_regularisation()
    ) :: PEVLM
    
    return PEVLM(;
        reference_wing_geometry=wing,
        kinematics=kinematics,
        dt=dt,
        free_stream_vel=free_stream_vel,
        current_time=current_time,
        wing_discretisation_chordwise=wing_discretisation_chordwise,
        wing_discretisation_spanwise=wing_discretisation_spanwise,
        regularisation_kernel=regularisation_kernel,
        regularisation_radius=regularisation_radius
    )
end

function advance_one_step!(a::PEVLM) :: Nothing
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
        throw(ErrorException("Probably blown up!"))
    end
    return
end


function convect_wake!(a::PEVLM) :: Nothing
    check(a.wing_lattice; do_assert=true)
    check(a.te_wake; do_assert=true)

    function get_dpoints()
        return get_vertices(a.te_wake)
    end
    function get_dvorts()
        locs, vorts = get_vortex_particles(a.te_wake)
        return vorts
    end
    function set_dpoints(new_te_points)
        set_vertices!(a.te_wake, new_te_points)
        return
    end
    function set_dvorts(new_vorts)
        a.te_wake.wake_particles.vorts = new_vorts
    end
    function vel_method()
        return field_velocity(a, get_vertices(a.te_wake);
            regularise_filaments=true)
    end
    function dvort_method() 
        p_locs, p_vorts = get_vortex_particles(a.te_wake)
        p_locse, p_vortse = everything_as_particles(a)
        p_dvorts = CVortex.particle_induced_dvort(
            p_locse, p_vortse, 
            p_locs, p_vorts,
            a.regularisation_kernel, a.regularisation_radius)  
        return p_dvorts
    end

    ode = PointsVortsODE(
        get_dpoints, get_dvorts, vel_method, dvort_method,
        set_dpoints, set_dvorts )
    runge_kutta_2_step(ode, a.dt)
    return
end


function update_wing_lattice!(a::PEVLM) :: Nothing
    z_offset = z_pos(a.kinematics, a.current_time)
    aoa = -AoA(a.kinematics, a.current_time)
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
    a.te_wake.lattice_buffer.strengths[:, end] = -a.wing_lattice.strengths[:, end]
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

    # Get const influences from the near_wake and free stream.
    ext_vel = nonwing_ind_vel(a, centres)

    # Get the velocity of the points on the wing.
    z_vel = dzdt(a.kinematics, a.current_time) 
    z_off = z_pos(a.kinematics, a.current_time)
    daoadt = -dAoAdt(a.kinematics, a.current_time)
    piv_x = pivot_position(a.kinematics)
    wing_vel = zeros(size(centres))
    wing_vel[:,3] .= z_vel
    wing_vel[:,1] += daoadt .* (centres[:,3] .- z_off)
    wing_vel[:,3] += daoadt .* (centres[:,1] .- piv_x)

    # Get the normal velocity from the wing / ext vels.
    ext_inf = sum((-ext_vel+wing_vel) .* normals, dims=2) # Dot product of each row.

    # Yay, finally something we can solve
    ring_strengths = inf_mat \ ext_inf

    rs = map(i->ring_strengths[i], wing_ring_idxs)
    a.wing_lattice.strengths = rs
    if !all(isfinite.(rs))
        @warn("Non-finite wing vorticities computed.")
    end
    return
end


function initialise_wake!(a::PEVLM) :: Nothing
    ni, nj = size(a.wing_lattice.strengths);
    verts = a.wing_lattice.vertices[:,nj:nj,:]
    a.te_wake.lattice_buffer = VortexLattice(verts)
    a.te_wake.edge_fil_strs = zeros(ni)
    return
end


function nonwing_ind_vel(
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
    fstarts, fends, fstrs = get_filaments( a.te_wake )
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

function everything_as_particles(
    a::PEVLM) :: Tuple{Matrix{Float32}, Matrix{Float32}}

    # Get vorticity sources.
    fwi_starts, fwi_ends, fwi_str = to_filaments(a.wing_lattice)
    fwa_starts, fwa_ends, fwa_str = get_filaments(a.te_wake)
    f_starts = cat(fwi_starts, fwa_starts; dims=1)
    f_ends = cat(fwi_ends, fwa_ends; dims=1)
    f_str = cat(fwi_str, fwa_str; dims=1)
    # Turn the filaments into particles so that our problem
    # doesn't explode.
    p_locs_f, p_vorts_f = to_vortex_particles( VortexFilament,
        f_starts, f_ends, f_str, a.regularisation_radius )
    p_locs_te, p_vorts_te = get_vortex_particles(a.te_wake)
    p_locs = cat(p_locs_te, p_locs_f; dims=1)
    p_vorts = cat(p_vorts_te, p_vorts_f; dims=1)
    return p_locs, p_vorts
end

function field_velocity(
    a::PEVLM,
    points :: Matrix{Float64};
    regularise_filaments=false
    ) :: Matrix{Float64}
    @assert(size(points)[2] == 3)
    @assert(all(isfinite.(points)), "Non-finite input point(s).")

    if regularise_filaments
        p_locs, p_vorts = everything_as_particles(a)
        @assert(all(isfinite.(p_vorts)))
        vels = CVortex.particle_induced_velocity(
            p_locs, p_vorts, points,
            a.regularisation_kernel, a.regularisation_radius)
        @assert(all(isfinite.(vels)))
    else
        # Get vorticity sources.
        fwi_starts, fwi_ends, fwi_str = to_filaments(a.wing_lattice)
        fwa_starts, fwa_ends, fwa_str = get_filaments(a.te_wake)
        f_starts = cat(fwi_starts, fwa_starts; dims=1)
        f_ends = cat(fwi_ends, fwa_ends; dims=1)
        f_str = cat(fwi_str, fwa_str; dims=1)
        p_locs, p_vorts = get_vortex_particles(a.te_wake)
        @assert(all(isfinite.(p_vorts)))

        # Get near wake lattice vel.
        vels = filament_induced_velocity(f_starts, f_ends, f_str, points)
        @assert(all(isfinite.(vels)))
        vels += CVortex.particle_induced_velocity(
            p_locs, p_vorts, points,
            a.regularisation_kernel, a.regularisation_radius)
        @assert(all(isfinite.(vels)))
    end
    vels[:,1] .+= a.free_stream_vel
    @assert(all(isfinite.(vels)))

    return vels
end

function field_vorticity(
    a::PEVLM,
    mes_locs::Matrix{Float32}) :: Matrix{Float32}

    p_locs, p_vorts = everything_as_particles(a)
    @assert(all(isfinite.(p_vorts)))

    vorts = CVortex.particle_field_vorticity(
        p_locs, p_vorts, mes_locs,
        a.regularisation_kernel, a.regularisation_radius)
    @assert(all(isfinite.(vorts)))
    return vorts
end

function redistribute_wake!(a::PEVLM)
    p_locs, p_vorts = get_vortex_particles(a.te_wake)
    nppos, npvort, ~ = CVortex.redistribute_particles_on_grid(
        p_locs, p_vorts, 
        CVortex.lambda3_redistribution(),
        a.regularisation_radius * (1/1.5);
        negligible_vort=0.25)
    a.te_wake.wake_particles.positions = nppos;
    a.te_wake.wake_particles.vorts = npvort
    return;
end

function relax_wake!(a::PEVLM; 
    relaxation_parameter::Float64=0.3) :: Nothing
    p_locs, p_vorts = get_vortex_particles(a.te_wake)
    npvort= CVortex.particle_pedrizzetti_relaxation(
        p_locs, p_vorts, relaxation_parameter,
        a.regularisation_kernel, a.regularisation_radius )
    a.te_wake.wake_particles.vorts = npvort
    return;
end

function a0_to_le_vort_const(a::PEVLM) :: Vector{Float64}
    # Gamma_1 = A_0 * THIS 
    wds = a.wing_discretisation_spanwise
    wdc = a.wing_discretisation_chordwise
    ret = zeros(length(wds)-1)
    le_verts = a.reference_wing_lattice.vertices[:,1,:]
    le_fil_centres = (le_verts[2:end,:] + le_verts[1:end-1,:]) / 2
    chords = a.reference_wing_geometry.chord_fn.(le_fil_centres[:,2])
    dtheta = (-acos(wdc[2]) - -acos(wdc[1]))
    corr = a.free_stream_vel * chords * 
        (sin(dtheta) + dtheta)
    return corr
end


function a0_value(a::PEVLM) :: Vector{Float64}
    str = a.wing_lattice.strengths[:,1]
    corr = a0_to_le_vort_const(a)
    a0 = str ./ corr
    return a0
end

function pressure_distribution(a::PEVLM; density::Float64=1.) :: Matrix{Float64}
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
    daoadt = -dAoAdt(a.kinematics, a.current_time)
    piv_x = pivot_position(a.kinematics)
    wing_vel = zeros(size(rc_vec))
    wing_vel[:,3] .= z_vel
    wing_vel[:,1] += daoadt .* (rc_vec[:,3] .- z_off)
    wing_vel[:,3] += daoadt .* (rc_vec[:,1] .- piv_x)
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

function lift_coefficient(a::PEVLM; lift_direction::Vector{<:Real}=[0.,0.,1.]) :: Float64
    @assert(length(lift_direction)==3)
    pressures = pressure_distribution(a)
    areas = ring_areas(a.wing_lattice)
    normal_force = pressures .* areas
    normals = ring_normals(a.wing_lattice) # Unit vectors
    ni, nj = size(areas)
    coeffs = zeros(ni, nj)
    for i = 1:ni
        for j = 1:nj
            coeffs[i,j] = sum(normals[i,j,:] .* lift_direction)
        end
    end
    lift = sum(normal_force .* coeffs)
    area = sum(ring_areas(a.wing_lattice))
    U = a.free_stream_vel
    CL = lift / (0.5 * U^2 * area)
    return CL
end

function lift_coefficient_distribution(a::PEVLM; 
        lift_direction::Vector{<:Real}=[0.,0.,1.]
        ) :: Tuple{Vector{Float64}, Vector{Float64}}

    @assert(length(lift_direction)==3)
    pressures = pressure_distribution(a)
    areas = ring_areas(a.wing_lattice)
    normal_force = pressures .* areas
    normals = ring_normals(a.wing_lattice) # Unit vectors
    centres = ring_centres(a.wing_lattice)
    ni, nj = size(areas)
    coeffs = zeros(ni, nj)
    for i = 1:ni
        for j = 1:nj
            coeffs[i,j] = sum(normals[i,j,:] .* lift_direction)
        end
    end
    lifts = normal_force .* coeffs
    areas = ring_areas(a.wing_lattice)
    # chord wise section basis. 
    lifts = map(x->sum(lifts[x,:]), collect(1:ni))
    areas = map(x->sum(areas[x,:]), collect(1:ni))

    U = a.free_stream_vel
    Cl = lifts ./ (0.5 * U^2 * areas)
    return Cl, centres[:,1,2]
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

function cross_section_vels(a::PEVLM, filename::String, y::Float64)
    xs = collect(-1:0.025:2)
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

function export_vorticity_field2(
    a::PEVLM, 
    filename::String;
    resolution::Float64=-1.0,
    bounds=[]) :: Nothing

    if resolution == -1.0
        resolution = a.regularisation_radius / 3
    end
    if length(bounds) == 0
        pts = a.te_wake.wake_particles.positions
        pts = vcat(pts, get_vertices(a.wing_lattice))
        bounds = [
            minimum(pts[:,1]), maximum(pts[:,1]),
            minimum(pts[:,2]), maximum(pts[:,2]),
            minimum(pts[:,3]), maximum(pts[:,3])]
    end
    @assert(resolution > 0)
    @assert(length(bounds) == 6)
    @assert(all(isfinite.(bounds)))

    xs = collect(bounds[1]:resolution:bounds[2])
    ys = collect(bounds[3]:resolution:bounds[4])
    zs = collect(bounds[5]:resolution:bounds[6])
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

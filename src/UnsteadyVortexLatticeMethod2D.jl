#
# UnsteadyVortexLatticeMethod2D.jl
#
# A 2D Unsteady Vortex Lattice Method
#
# Copyright HJAB 2020
#
################################################################################

mutable struct UnsteadyVortexLatticeMethod2D

    free_stream_vel :: Float64
    kinematics :: RigidKinematics2D
    reference_foil_lattice :: ParticleGroup2D # TE at jmax.

    foil_lattice :: ParticleGroup2D       
    old_foil_lattice :: ParticleGroup2D
    te_wake :: ParticleGroup2D
    le_wake :: ParticleGroup2D

    dt :: Float64
    current_time :: Float64

    critical_le_vorticity ::Float64
    le_shed_on_last_step :: Bool

    regularisation_radius :: Float64
    regularisation_kernel :: CVortex.RegularisationFunction
    initialised :: Bool

    # Detailed constructor
    function UnsteadyVortexLatticeMethod2D(;
        reference_foil_lattice :: ParticleGroup2D,
        kinematics :: RigidKinematics2D,
        dt :: Float64 = -1.,
        free_stream_vel :: Real = 1.,
        current_time :: Float64 = 0.,
        critical_le_vorticity :: Float64 = 9e99,
        regularisation_radius :: Float64 = -1.,
        regularisation_kernel :: CVortex.RegularisationFunction = singular_regularisation())

        szw = size(reference_foil_lattice.positions)
        foil_lattice = ParticleGroup2D()
        old_foil_lattice = ParticleGroup2D()
        te_wake = ParticleGroup2D()
        le_wake = ParticleGroup2D()

        new(free_stream_vel, kinematics,
            reference_foil_lattice,
            foil_lattice, old_foil_lattice, te_wake, le_wake,
            dt, current_time, critical_le_vorticity, false,
            regularisation_radius, regularisation_kernel, false)
    end
end

function UnsteadyVortexLatticeMethod2D(kinematics ::RigidKinematics2D;
    free_stream_vel :: Real = 1.0, 
    foil :: ThinFoilGeometry = ThinFoilGeometry(0.5,x->0),
    regularisation_kernel :: CVortex.RegularisationFunction = singular_regularisation(), 
    regularisation_radius :: Real = -1, 
    dt :: Real = -1,
    wing_discretisation_chordwise::Vector{<:Real}=collect(range(-1, 1; length=40)),
    current_time::Real = 0,
    a0_critical = -1
    ) :: UnsteadyVortexLatticeMethod2D

    semichord = foil.semichord
    ref_lattice = ParticleGroup2D()
    wdc = wing_discretisation_chordwise       # The ring LE at TE are set quarter of a ring back.
    cwdc = deepcopy(wdc)
    cwdc[1:end-1] = wdc[1:end-1] + 0.25 * (wdc[2:end]-wdc[1:end-1])
    cwdc[end] = wdc[end] + 0.25 * (wdc[end] - wdc[end-1])
    ref_pos = zeros(length(wing_discretisation_chordwise), 2)
    ref_pos[1:end,1] = cwdc .* foil.semichord
    ref_pos[1:end-1,2] = foil.camber_line.(cwdc[1:end-1]) .* semichord
    ref_pos[end,2] = 0.

    if a0_critical == -1
        critical_le_vorticity = 9e99
    else
        dx1 = wdc[2] - wdc[1]
        corr = free_stream_vel * 2 * semichord *(
            acos(1 - (dx1 / semichord)) +
            sin(acos(1 - dx1 / semichord)))
        critical_le_vorticity = corr * a0_critical
    end

    if dt == -1
        dt = 0.05 * semichord / free_stream_vel
    end
    ref_lattice.positions = ref_pos
    ref_lattice.vorts = zeros(size(ref_pos)[1])
    @assert(dt != 0.0)
    regularisation_radius = dt * free_stream_vel

    return UnsteadyVortexLatticeMethod2D(
        reference_foil_lattice=ref_lattice,
        kinematics=kinematics,
        dt=dt,
        free_stream_vel=free_stream_vel,
        current_time=Float64(current_time),
        critical_le_vorticity = critical_le_vorticity,
        regularisation_kernel=regularisation_kernel,
        regularisation_radius=regularisation_radius)
end

function advance_one_step(a::UnsteadyVortexLatticeMethod2D) :: Nothing
    if !a.initialised
        update_foil_lattice!(a)
        initialise_wake!(a)
        a.initialised = true
    end

    a.current_time += a.dt
    convect_wake!(a)
    a.old_foil_lattice = a.foil_lattice
    update_foil_lattice!(a)
    update_wake_lattice!(a)
    compute_foil_vortex_strengths!(a)
    return
end



function convect_wake!(a::UnsteadyVortexLatticeMethod2D) :: Nothing
    check(a.te_wake; do_assert=true)

    points = cat(a.te_wake.positions, a.le_wake.positions[1:end-1,:]; dims=1)
    ntep = size(a.te_wake.positions)[1]
    nlep = size(a.le_wake.positions[1:end-1,:])[1]
    vels = field_velocity(a, points)
    dt = a.dt
    new_points = points + vels .* dt
    a.te_wake.positions = new_points[1:ntep,:]
    a.le_wake.positions[1:end-1,:] = new_points[ntep+1:end,:]
    return
end


function update_foil_lattice!(a::UnsteadyVortexLatticeMethod2D) :: Nothing
    z_offset = z_pos(a.kinematics, a.current_time)
    aoa = AoA(a.kinematics, a.current_time)
    piv_x = pivot_position(a.kinematics)
    
    new_geometry = deepcopy(a.reference_foil_lattice.positions)
    dx = new_geometry[:,1] .- piv_x
    new_geometry[:,1] = cos(aoa) * dx .+ piv_x
    new_geometry[:,2] = sin(aoa) * dx
    new_geometry[:,2] .+= z_offset
    a.foil_lattice.positions = new_geometry

    if length(a.foil_lattice.vorts)==0
        a.foil_lattice.vorts = zeros(size(new_geometry)[1])
    end
    return
end


function update_wake_lattice!(a::UnsteadyVortexLatticeMethod2D) :: Nothing
    update_te_wake_lattice!(a)
    insert_le_wake_particle!(a)
    return
end 


function update_te_wake_lattice!(a::UnsteadyVortexLatticeMethod2D) :: Nothing
    check(a.te_wake; do_assert=true)

    # We need to add a new row to the lattice and reconnect it to the wing.
    foil_rings = ring_strengths_from_vorts(UnsteadyVortexLatticeMethod2D, a.foil_lattice)

    a.te_wake.positions = cat(
        a.te_wake.positions, 
        a.foil_lattice.positions[end:end,:]; 
        dims=1)    
    a.te_wake.vorts = cat(
        a.te_wake.vorts, 
        [0]; 
        dims=1) # Just needs to be right length
    
    te_wake_rings = ring_strengths_from_vorts(UnsteadyVortexLatticeMethod2D, a.te_wake)
    te_wake_rings = cat(
        te_wake_rings[1:end-1], 
        -foil_rings[end]; 
        dims=1)
    te_wake_vorts = vorts_from_ring_strengths(UnsteadyVortexLatticeMethod2D, te_wake_rings)
    @assert(length(te_wake_vorts)==length(a.te_wake.vorts))
    a.te_wake.vorts = te_wake_vorts
    check(a.te_wake; do_assert=true)
    return
end


function insert_le_wake_particle!(a::UnsteadyVortexLatticeMethod2D) :: Nothing
    check(a.le_wake; do_assert=true)

    le_pos = foil_le_position(UnsteadyVortexLatticeMethod2D, a.foil_lattice)
    lp_pos = a.le_wake.positions[end-1,:]
    newpos = le_pos + (lp_pos-le_pos) ./ 3

    a.le_wake.positions = cat(
        a.le_wake.positions[1:end-1,:], 
        reshape(newpos,1,:),
        reshape(a.foil_lattice.positions[1,:],1,:); # We want to keep the lattice edge on the leading edge fil.
        dims=1)    
    a.le_wake.vorts = cat(
        a.le_wake.vorts, 
        [0]; 
        dims=1)
    le_wake_rings = ring_strengths_from_vorts(UnsteadyVortexLatticeMethod2D, a.le_wake)
    le_wake_rings = cat(
        le_wake_rings[1:end-1], 
        0; # New wake wing strength must be calculated simultaniously w/ foil ring strs.
        dims=1)
    le_wake_vorts = vorts_from_ring_strengths(UnsteadyVortexLatticeMethod2D, le_wake_rings)
    @assert(length(le_wake_vorts)==length(a.le_wake.vorts))
    a.le_wake.vorts = le_wake_vorts
    check(a.le_wake; do_assert=true)
    return
end


function initialise_wake!(a::UnsteadyVortexLatticeMethod2D) :: Nothing 
    a.te_wake = ParticleGroup2D()
    a.te_wake.positions = a.foil_lattice.positions[end:end,:]
    a.te_wake.vorts = [0.0]
    wvel = wing_surface_velocity(a, a.foil_lattice.positions[1:1,:])
    a.le_wake = ParticleGroup2D()
    a.le_wake.positions = cat(
        a.foil_lattice.positions[1:1,:] + reshape(
            a.dt * ([a.free_stream_vel,0.] - reshape(wvel,:)) * 1.5, 1, :),
        a.foil_lattice.positions[1:1,:];
        dims=1)
    a.le_wake.vorts = [0.0, 0.0]
    return
end


function wing_surface_velocity(
    a::UnsteadyVortexLatticeMethod2D,
    points_on_wing::Matrix{<:Real}) :: Matrix{Float64}

    @assert(size(points_on_wing)[2]==2)

    # Get the velocity of the points on the wing.
    z_vel = dzdt(a.kinematics, a.current_time) 
    z_off = z_pos(a.kinematics, a.current_time)
    daoadt = dAoAdt(a.kinematics, a.current_time)
    piv_x = pivot_position(a.kinematics)
    wing_vel = zeros(size(points_on_wing))
    wing_vel[:,2] .= z_vel
    wing_vel[:,1] += daoadt .* (points_on_wing[:,2] .- z_off)
    wing_vel[:,2] += daoadt .* (points_on_wing[:,1] .- piv_x)
    return wing_vel
end


function wing_self_influence_matrix(
    a::UnsteadyVortexLatticeMethod2D,
    points_on_wing::Matrix{Float64},
    normals_on_wing::Matrix{Float64}) :: Matrix{Float64}

    @assert(size(points_on_wing)[2]==2)
    @assert(size(normals_on_wing)==size(points_on_wing))

    infs = zeros(size(points_on_wing)[1], size(points_on_wing)[1])
    rrad = a.regularisation_radius
    kern = a.regularisation_kernel
    vel1 = zeros(size(points_on_wing))
    vel2 = zeros(size(points_on_wing))
    velfn = posn->mapreduce(
        i->particle_induced_velocity(posn, 1, points_on_wing[i,:], 
        CVortex.singular_regularisation(), 1)', 
        #a.regularisation_kernel, 1)', 
        vcat, 1:size(points_on_wing)[1])
    vel2 = velfn(a.foil_lattice.positions[1,:])
    for i = 1 : size(points_on_wing)[1]
        vel1 = vel2
        vel2 = velfn(a.foil_lattice.positions[i+1,:])
        vel = vel1 - vel2
        ndot = sum(normals_on_wing .* vel; dims=2)
        infs[i,:] = ndot
    end
    return infs
end


function wing_centres_and_normals(
    a::UnsteadyVortexLatticeMethod2D
    ) :: Tuple{Matrix{Float64}, Matrix{Float64}}

    centres = a.foil_lattice.positions
    normals = centres[2:end,:] .- centres[1:end-1, :]
    for i = 1 : size(normals)[1]
        normals[i,:] /= sqrt(normals[i,1]^2 + normals[i,2]^2)
        normals[i,:] = [-normals[i,2], normals[i,1]]
    end
    centres = (centres[1:end-1,:] .+ centres[2:end,:]) / 2
    return centres, normals
end


function compute_lev_influence_matrix(
    a::UnsteadyVortexLatticeMethod2D,
    points_on_wing::Matrix{Float64},
    normals_on_wing::Matrix{Float64}) :: Matrix{Float64}
    @assert(size(points_on_wing)[2]==2)
    @assert(size(normals_on_wing)==size(points_on_wing))

    rrad = a.regularisation_radius
    kern = a.regularisation_kernel
    vel1 = zeros(size(points_on_wing))
    vel2 = zeros(size(points_on_wing))
    velfn = posn->mapreduce(
        i->particle_induced_velocity(posn, 1, points_on_wing[i,:], 
        CVortex.singular_regularisation(), 1)', 
        #a.regularisation_kernel, 1)', 
        vcat, 1:size(points_on_wing)[1])
    vel1 = velfn(a.le_wake.positions[end-1,:])
    vel2 = velfn(a.le_wake.positions[end,:])
    vel = vel1 - vel2
    ndot = sum(normals_on_wing .* vel; dims=2)
    infs = ndot
    return infs
end


function compute_foil_vortex_strengths!(
    a::UnsteadyVortexLatticeMethod2D; 
    shedding_le::Bool = false # shedding_le = true should only be called from inside fn.
    ) :: Nothing

    centres, normals = wing_centres_and_normals(a)
    r_strs = ring_strengths_from_vorts(UnsteadyVortexLatticeMethod2D, a.foil_lattice)

    # The wing's self influence
    infs = wing_self_influence_matrix(a, centres, normals)
    if shedding_le
        # Were using the formulation Gamma_LER = Gamma_1 -+ Gamma_LESP
        le_sgn = a.foil_lattice.vorts[1] > 0 ? 1. : -1.
        lev_inf = compute_lev_influence_matrix(a, centres, normals)
        infs[:,1] += lev_inf
    end

    # Get const influences from the te_wake and free stream.
    ext_vel = external_induced_vel(a, centres)
    wing_vel = wing_surface_velocity(a, centres)
    lesp_const_vel = zeros(size(ext_vel)[1])   # Zero unless over lesp crit!
    if shedding_le
        lesp_const_vel = -le_sgn * a.critical_le_vorticity * lev_inf
    end
    # Take dot product of velocities with normals. 
    ext_inf = sum((wing_vel-ext_vel) .* normals, dims=2)-lesp_const_vel

    # Yay, finally something we can solve
    ring_strengths = reshape(infs \ ext_inf, :)

    if shedding_le
        vorts = vorts_from_ring_strengths(
            UnsteadyVortexLatticeMethod2D, ring_strengths)
        a.foil_lattice.vorts = vorts
        rs_levs = ring_strengths_from_vorts(
            UnsteadyVortexLatticeMethod2D,  a.le_wake)
        rs_levs[end] = ring_strengths[1] - le_sgn * a.critical_le_vorticity
        vorts_lew = vorts_from_ring_strengths(
            UnsteadyVortexLatticeMethod2D, rs_levs)
        a.le_wake.vorts = vorts_lew
        a.le_shed_on_last_step = true
    else
        vorts = vorts_from_ring_strengths(
            UnsteadyVortexLatticeMethod2D, ring_strengths)
        a.foil_lattice.vorts = vorts    # We have to set this
                                        # so we know to use +ve or -ve
                                        # LE vort.
        if abs(vorts[1]) > a.critical_le_vorticity 
            compute_foil_vortex_strengths!(a; shedding_le=true)
        else
            a.le_shed_on_last_step = false
        end
    end

    if !shedding_le
        vels = field_velocity(a, centres)
    end

    return
end


function ring_strengths_from_vorts(
    ::Type{UnsteadyVortexLatticeMethod2D},
    a::ParticleGroup2D;
    do_assert::Bool=true) :: Vector{Float64}

    check(a; do_assert=true)

    strs = zeros(length(a.vorts)-1)
    lstrs = length(strs)
    for i = 1:lstrs
        strs[i] = a.vorts[i]
        if i > 1
            strs[i] += strs[i-1]
        end
    end
    if do_assert
        err = a.vorts[end] - strs[end]
        relerr = err / strs[end]
        @assert((err==0) || relerr < 0.0001, "Set of vortexes does not "*
            "have net zero circulation. Expected "*
            string(-strs[end])*" but got "*string(a.vorts[end])*".")
    end
    return strs
end


function vorts_from_ring_strengths(
    ::Type{UnsteadyVortexLatticeMethod2D},
    strs::Vector{<:Real}) :: Vector{Float64}

    @assert(all(isfinite.(strs)))
    vorts = zeros(length(strs)+1)
    vorts[1:end-1] = strs
    vorts[2:end] -= strs
    return vorts
end


function external_induced_vel(
    a::UnsteadyVortexLatticeMethod2D,
    points :: Matrix{<:Real}
    ) :: Matrix{Float64}

    @assert(size(points)[2]==2)
    check(a.te_wake; do_assert=true)
    check(a.le_wake; do_assert=true)
    # Get const influences from the te_wake and free stream.
    ext_vel = zeros(size(points))
    ext_vel[:,1] .= a.free_stream_vel
    nwi = length(a.te_wake.vorts)
    wake_ind_vel = CVortex.particle_induced_velocity(
        cat(a.te_wake.positions, a.le_wake.positions; dims=1), 
        cat(a.te_wake.vorts, a.le_wake.vorts; dims=1),
        points, a.regularisation_kernel,
        a.regularisation_radius )
    ext_vel += wake_ind_vel
    return ext_vel
end


function field_velocity(
    a::UnsteadyVortexLatticeMethod2D,
    points::Matrix{<:Real}) :: Matrix{Float32}

    @assert(size(points)[2]==2)
    vels = CVortex.particle_induced_velocity(
        cat(a.te_wake.positions, a.le_wake.positions; dims=1), 
        cat(a.te_wake.vorts, a.le_wake.vorts; dims=1),
        points, a.regularisation_kernel,
        a.regularisation_radius )
    vels += particle_induced_velocity(a.foil_lattice.positions, 
        a.foil_lattice.vorts, points, 
        #CVortex.singular_regularisation(), a.regularisation_radius)
        a.regularisation_kernel, a.regularisation_radius)
    vels[:,1] .+= a.free_stream_vel
    return vels
end


function to_vtk(a::UnsteadyVortexLatticeMethod2D,
    filename_start::String, file_no::Int;
    translation::Vector{<:Real}=[0,0,0]) :: Nothing

    f_foil = filename_start * "_foil_" * string(file_no)
    f_te_wake = filename_start * "_te_wake_" * string(file_no)
    f_le_wake = filename_start * "_le_wake_" * string(file_no)
    to_vtk(a.foil_lattice, f_foil)
    to_vtk(a.te_wake, f_te_wake)
    to_vtk(a.le_wake, f_le_wake)
    return
end


function cross_section_vels(a::UnsteadyVortexLatticeMethod2D, filename::String)
    xs = collect(-1:0.01:1)
    ys = collect(-1:0.01:1)
    nx = length(xs)
    ny = length(ys)
    coord_msh = zeros(nx, ny, 2)
    for ix = 1 : nx
        for iz = 1 : ny
            coord_msh[ix,iz,:] = [xs[ix], ys[iz]]
        end
    end
    coord_msh = reshape(coord_msh,:,2)
    vels = field_velocity(a, coord_msh)
    idxs = collect(1 : size(vels)[1])
    idxs = reshape(idxs, nx, ny)

    points = coord_msh
    ncells = (nx-1)*(ny-1)
    cells = Vector{WriteVTK.MeshCell}(undef, ncells)
    acc = 0
    for ix = 1 : nx-1
        for iy = 1 : ny-1
            acc += 1
            cells[acc] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUAD, 
                [idxs[ix,iy], idxs[ix,iy+1], idxs[ix+1,iy+1], idxs[ix+1,iy]])
        end
    end

    p3ds = cat(points, zeros(size(points)[1]); dims=2)
    v3ds = cat(vels, zeros(size(vels)[1]); dims=2)
    vtkfile = WriteVTK.vtk_grid(filename, p3ds', cells)
    WriteVTK.vtk_point_data(vtkfile, v3ds', "Velocity")
    WriteVTK.vtk_save(vtkfile)
end


function foil_le_position(
    ::Type{UnsteadyVortexLatticeMethod2D},
    foil_lattice :: ParticleGroup2D) :: Vector{Float64}

    posns = foil_lattice.positions
    p0 = posns[1,:]
    p1 = posns[2,:]
    ple = p0 - 0.25 * (p1 - p0)
    return ple
end


function foil_te_position(
    ::Type{UnsteadyVortexLatticeMethod2D},
    foil_lattice :: ParticleGroup2D) :: Vector{Float64}

    posns = foil_lattice.positions
    pnm1 = posns[end-1,:]
    pn = posns[end,:]
    pte= pnm1 + 0.75 * (pn-pnm1)
    return pte
end


function total_field_vorticity(
    a::UnsteadyVortexLatticeMethod2D) :: Float64
    return (total_vorticity(a.foil_lattice) 
        + total_vorticity(a.te_wake)
        + total_vorticity(a.le_wake))
end

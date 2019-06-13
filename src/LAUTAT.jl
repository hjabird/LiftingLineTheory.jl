
mutable struct ThinFoilGeometry
    semichord :: Real
    camber_line :: Function # In [-1, 1]
    camber_slope :: Function

    function ThinFoilGeometry()
        return new(0.5, x->0, x->0)
    end
    function ThinFoilGeometry(semichord::Real)
        @assert(semichord>0, "Semichord must be positive")
        return new(semichord, x->0, x->0)
    end
    function ThinFoilGeometry(semichord::Real, camber_func::Function)
        @assert(semichord>0, "Semichord must be positive")
        @assert(hasmethod(camber_func, (Float64,)), "Camber function "*
            "must accept single argument in [-1 (LE), 1 (TE)].")
        @assert(hasmethod(camber_func, (Real,)), "Camber function "*
            "must accept real arguments for automatic differentiation.")
        return new(semichord, camber_func, x->ForwardDiff.derivative(camber_func, x))
    end
end

mutable struct RigidKinematics
    z_pos :: Function
    dzdt :: Function
    AoA :: Function
    dAoAdt :: Function
    pivot_position :: Real
    function RigidKinematics(z::Function, AoA::Function, pivot_position)
        @assert(hasmethod(z, (Float64,)), "z function "*
            "must accept single argument of time.")
        @assert(hasmethod(z, (Real,)), "z function "*
            "must accept real arguments for automatic differentiation.")
        @assert(hasmethod(AoA, (Float64,)),  "AoA function "*
        "must accept single argument of time.")
        @assert(hasmethod(AoA, (Real,)), "AoA function "*
            "must accept real arguments for automatic differentiation.")
        return new(
            z, x->ForwardDiff.derivative(z, x), 
            AoA, x->ForwardDiff.derivative(AoA, x), pivot_position)
    end
end

mutable struct ParticleGroup2D
    positions :: Matrix{Float32} # An N by 3 Matrix
    vorts :: Vector{Float32}
    function ParticleGroup2D()
        return new(zeros(Float32, 0, 2), zeros(Float32, 0))
    end
end

mutable struct LAUTAT
    U :: Vector{Real} # Free stream velocity
    kinematics :: RigidKinematics

    foil :: ThinFoilGeometry
    te_particles :: ParticleGroup2D
    regularisation :: RegularisationFunction
    reg_dist :: Real

    num_fourier_terms :: Integer
    current_fourier_terms :: Vector{Real}
    last_fourier_terms :: Vector{Real}
    current_time :: Real
    dt :: Real

    function LAUTAT(;U=[1.,0], foil=ThinFoilGeometry(0.5,x->0),
        kinematics=RigidKinematics(x->x, x->0, 0.0), te_particles=ParticleGroup2D(),
        regularisation=winckelmans_regularisation(), reg_dist_factor=1.5,
        num_fourier_terms=8, current_fourier_terms=[], last_fourier_terms=[],
        current_time=0.0, dt=0.025)

        return new(U, kinematics, foil, te_particles, regularisation,
            sqrt(U[1]^2 + U[2]^2) * dt * 1.5, num_fourier_terms, 
            current_fourier_terms, last_fourier_terms, current_time, dt)
    end
end

function foil_points(a::LAUTAT, points::Vector{<:Real})
    @assert(all(-1 .<= points .<= 1), "All points must be in [-1,1]")
    x1 = zeros(length(points), 2)
    x1[:,1] = points .* a.foil.semichord .- a.kinematics.pivot_position
    x1[:,2] = a.foil.semichord .* map(a.foil.camber_line, points)
    AoA = a.kinematics.AoA(a.current_time)
    rot = [cos(-AoA) -sin(-AoA); sin(-AoA) cos(-AoA)]
    for i = 1 : length(points)
        x1[i, :] = rot * x1[i,:]
    end
    x1[:,1] .+= a.kinematics.pivot_position
    x1[:,2] .+= a.kinematics.z_pos(a.current_time)
    return x1
end

function bound_vorticity_density(a::LAUTAT, local_pos::Real)
    @assert(-1<local_pos<=1, "local position must be in (-1,1]")
    theta = acos(-local_pos)
    vd = local_pos==1 ? 0 : a.current_fourier_terms[1] * (
            1 + cos(theta)) / sin(theta)
    for i = 2:length(a.current_fourier_terms)
        vd += a.current_fourier_terms[i] * sin((i-1)* theta)
    end
    vd *= 2 * sqrt(a.U[1]^2 + a.U[2]^2)
    return vd
end

function bound_vorticity(a::LAUTAT)
    vort = a.foil.semichord * pi * (2 * a.current_fourier_terms[1] +
        a.current_fourier_terms[2]) * sqrt(a.U[1]^2 + a.U[2]^2)
    return vort
end

function foil_induced_vel(a::LAUTAT, mes_pnts::Matrix{<:Real})
    points, weights = FastGaussQuadrature.gausslegendre(50)
    vortex_pos = foil_points(a, points)
    weights .*= a.foil.semichord
    strengths = map(x->bound_vorticity_density(a, x), points)
    kernel = winckelmans_regularisation()
    vels = particle_induced_velocity(vortex_pos, strengths, mes_pnts, 
        kernel, a.reg_dist)    
    return vels
end

function te_wake_particle_velocities(a::LAUTAT)
    reg_dist = a.reg_dist
    kernel = winckelmans_regularisation()
    vel_self = particle_induced_velocity(a.te_particles.positions,
        a.te_particles.vorts, a.te_particles.positions, a.regularisation, reg_dist)
    vel_foil = foil_induced_vel(a, a.te_particles.positions)
    vels = vel_self + vel_foil
    vels[:, 1] .+= a.U[1]
    vels[:, 2] .+= a.U[2]
    return vels
end

function vel_normal_to_foil_surface(a::LAUTAT, mes_pnts::Vector{<:Real})
    @assert(all(-1 .<= mes_pnts .<= 1), "Foil in [-1,1]")
    fpoints = foil_points(a, mes_pnts)
    wake_vels = particle_induced_velocity(a.te_particles.positions, 
        a.te_particles.vorts, fpoints, a.regularisation, a.reg_dist)
    ext_vels = a.U
    alpha = a.kinematics.AoA(a.current_time)
    alpha_dot = a.kinematics.dAoAdt(a.current_time)
    dzdt = a.kinematics.dzdt(a.current_time)
    slopes= map(a.foil.camber_slope, mes_pnts)
    rot = [cos(alpha) -sin(alpha); sin(alpha) cos(alpha)]
    ext_vels = rot * ext_vels
    for i = 1 : length(mes_pnts)
        wake_vels[i, :] = rot * wake_vels[i, :]
    end
    wash = (slopes .* (ext_vels[1] .+ dzdt * sin(alpha) .+ wake_vels[:, 1])
        .- ext_vels[2]
        .- alpha_dot * a.foil.semichord .* (mes_pnts .- a.kinematics.pivot_position)
        .+ dzdt * cos(alpha) .- wake_vels[:, 2])
    return wash
end

function compute_fourier_terms(a::LAUTAT)
    points, weights = FastGaussQuadrature.gausslegendre(30)
    points, weights = linear_remap(points, weights, -1, 1, 0, pi)
    dwsh = vel_normal_to_foil_surface(a, -cos.(points))
    fterms = zeros(a.num_fourier_terms)
    for i = 1 : a.num_fourier_terms
        qpoints = cos.((i-1)*points) .* dwsh * 2 /(sqrt(a.U[1]^2 + a.U[2]^2) * pi)
        fterms[i] = sum(qpoints .* weights)
    end
    fterms[1] /= -2
    return fterms
end

function pivot_coordinate(foil::ThinFoilGeometry, kinem::RigidKinematics, t::Real)
    pos = [0., 0.]
    pos[1] = kinem.pivot_position * foil.semichord
    pos[2] = kinem.z_pos(t)
    return pos
end

function foil_velocity(a::LAUTAT, local_pos::Vector{<:Real})
    @assert(all(-1 .<= local_pos .<= 1))
    angular_vel = a.kinematics.dAoAdt(a.current_time)
    radii = foil_points(a, local_pos) - pivot_coordinate(a.foil, a.kinematics, a.current_time)'
    vel = zeros(length(local_pos), 2)
    vel[:, 1] = -angular_vel .* radii[:, 1]
    vel[:, 2] = angular_vel .* radii[:, 2] .+ a.kinematics.dzdt(a.current_time)
    return vel
end

function shed_new_te_particle_with_zero_vorticity!(a::LAUTAT)
    @assert(size(a.te_particles.positions)[1] == length(a.te_particles.vorts))
    np = length(a.te_particles.vorts)
    if np == 0 # The first shed particle
        part_pos = foil_points(a, [1])[1,:]'
        vel = -foil_velocity(a, [1])
        vel .+= a.U'
        part_pos += vel * a.dt * 0.5
    else 
        part_pos = a.te_particles.positions[end,:]'
        te_coord = foil_points(a, [1])[1,:]'
        part_pos -= 2/3 * (part_pos  - te_coord)
    end
    a.te_particles.positions = vcat(a.te_particles.positions, part_pos)
    push!(a.te_particles.vorts, 0)
    return
end

function adjust_last_shed_te_particle_for_kelvin_condition!(a::LAUTAT)
    @assert(size(a.te_particles.positions)[1] == length(a.te_particles.vorts))
    alpha = a.kinematics.AoA(a.current_time)
    alpha_dot = a.kinematics.dAoAdt(a.current_time)
    dzdt = a.kinematics.dzdt(a.current_time)
    qpoints, qweights = FastGaussQuadrature.gausslegendre(50)
    qpoints, qweights = linear_remap(qpoints, qweights, -1, 1, 0, pi)
    # Compute the influence of the known part of the wake
    I_k = sum(
        (vel_normal_to_foil_surface(a, -cos.(qpoints)) .* (cos.(qpoints).-1) .* 
        2*a.foil.semichord) .* qweights)
    # And the bit that will be caused by the new particle
    posn = a.te_particles.positions[end,:]
    rot = [cos(alpha) -sin(alpha); sin(alpha) cos(alpha)]
    function I_uk_integrand(theta::Vector{<:Real})
        foil_pos = -cos.(theta)
        println("THetas: ", theta)
        foil_coords = foil_points(a, foil_pos)
        println("Coords: ", foil_coords)
        println("Vort pos: ", posn)
        vels = mapreduce(
            i->(rot * particle_induced_velocity(posn, 1., foil_coords[i,:], a.regularisation, a.reg_dist))',
            vcat, 1 : length(qpoints))
        println("Vels: ", vels)
        normal_vel = vels[:, 1].*map(a.foil.camber_slope, foil_pos) .- vels[:,2]
        return normal_vel .* (cos.(theta).-1) * 2. * a.foil.semichord
    end
    I_uk = sum(I_uk_integrand(qpoints) .* qweights)
    # And now work out the vorticity
    println("I_uk: ", I_uk)
    println("Ik ", I_k)
    vort = - (I_k + total_te_vorticity(a)) / (1 + I_uk)
    a.te_particles.vorts[end] = vort
    println("Vort p: ", vort)
    return
end

function total_te_vorticity(a::LAUTAT)
    return sum(a.te_particles.vorts)
end

function advance_one_step(a::LAUTAT)
    if(length(a.current_fourier_terms)==0)
        tmptime = a.current_time
        a.current_time -= a.dt
        a.current_fourier_terms = compute_fourier_terms(a)
        a.last_fourier_terms = a.current_fourier_terms
        a.current_time = tmptime
    end        
    wake_vels = te_wake_particle_velocities(a::LAUTAT)
    a.te_particles.positions += wake_vels .* a.dt
    a.current_time += a.dt
    shed_new_te_particle_with_zero_vorticity!(a)
    adjust_last_shed_te_particle_for_kelvin_condition!(a)
    a.current_fourier_terms = compute_fourier_terms(a)
    a.last_fourier_terms = a.current_fourier_terms
    return
end

function to_vtk(a::LAUTAT, filename::String)
    np = length(a.te_particles.vorts)
    cells = Vector{WriteVTK.MeshCell}(undef, np+29)
    points = zeros(np+30, 3)
    vorts = zeros(np+30)
    vorts[1:np] = a.te_particles.vorts
    for i = 1 : np
        cells[i] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_VERTEX, [i])
        points[i, :] = [
            a.te_particles.positions[i,1], 
            a.te_particles.positions[i,2], 0.]
    end
    localpos = collect(-1:2/29:1)
    points[np+1:end, :] = hcat(foil_points(a, localpos), zeros(30, 1))
    bv = vcat([NaN], map(x->bound_vorticity_density(a, x), localpos[2:end]))
    vorts[np+1:end] = bv .* a.foil.semichord/30
    for i = 1 : 29
        cells[i + np] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LINE, 
            [i + np, i + np + 1])
    end
    vtkfile = WriteVTK.vtk_grid(filename, points', cells)
    WriteVTK.vtk_point_data(vtkfile, vorts, "Vorticity")
    WriteVTK.vtk_save(vtkfile)
    return
end

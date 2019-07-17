using PyPlot

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
    function ThinFoilGeometry(semichord::Real, camber_func::Function,
        camber_slope::Function)
        @assert(semichord>0, "Semichord must be positive")
        @assert(hasmethod(camber_func, (Float64,)), "Camber function "*
            "must accept single argument in [-1 (LE), 1 (TE)].")
        @assert(hasmethod(camber_slope, (Float64,)), "Camber slope function "*
            "must accept single argument in [-1 (LE), 1 (TE)].")
        return new(semichord, camber_func, camber_slope)
    end
end

mutable struct RigidKinematics2D
    z_pos :: Function
    dzdt :: Function
    AoA :: Function
    dAoAdt :: Function
    pivot_position :: Real
    function RigidKinematics2D(z::Function, AoA::Function, pivot_position)
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
    external_perturbation :: Function #f(x::Matrix{Real}, t::Real) where x
    # is of size (N, 2) where N is number of mes pos and returns vec of (N, 2)
    kinematics :: RigidKinematics2D

    foil :: ThinFoilGeometry
    te_particles :: ParticleGroup2D
    regularisation :: RegularisationFunction
    reg_dist :: Real

    num_fourier_terms :: Integer
    current_fourier_terms :: Vector{Real}
    last_fourier_terms :: Vector{Real}
    current_time :: Real
    dt :: Real

    dw_positions :: Vector{Float64} # N points in [-1, 1]
    dw_weights :: Vector{Float64}   # N points for quadrature of DW.
    dw_values :: Matrix{Float64}    # (N, 2) induced velocities.

    function LAUTAT(;U=[1.,0]::Vector{<:Real}, 
        external_perturbation=(x,t)->zeros(size(x)[1], 2),
        foil=ThinFoilGeometry(0.5,x->0),
        kinematics=RigidKinematics2D(x->x, x->0, 0.0), te_particles=ParticleGroup2D(),
        regularisation=winckelmans_regularisation(), reg_dist_factor=1.5,
        num_fourier_terms=8, current_fourier_terms=[], last_fourier_terms=[],
        current_time=0.0, dt=0.025)

        return new(U, external_perturbation, kinematics, foil, te_particles, 
            regularisation, sqrt(U[1]^2 + U[2]^2) * dt * 1.5, num_fourier_terms, 
            current_fourier_terms, last_fourier_terms, current_time, dt,
            [], [], zeros(0,0))
    end
end

function advance_one_step(a::LAUTAT)
    if(length(a.current_fourier_terms)==0)
        fill_downwash_cache!(a, 50)
        tmptime = a.current_time
        a.current_time -= a.dt
        a.current_fourier_terms = compute_fourier_terms(a)
        a.last_fourier_terms = a.current_fourier_terms
        a.current_time = tmptime
        invalidate_downwash_cache!(a)
    end        
    wake_vels = te_wake_particle_velocities(a::LAUTAT)
    a.te_particles.positions += wake_vels .* a.dt
    invalidate_downwash_cache!(a)
    a.current_time += a.dt
    fill_downwash_cache!(a, 64)
    shed_new_te_particle_with_zero_vorticity!(a)
    adjust_last_shed_te_particle_for_kelvin_condition!(a)
    a.last_fourier_terms = a.current_fourier_terms
    a.current_fourier_terms = compute_fourier_terms(a)
    return
end

function fill_downwash_cache!(a::LAUTAT, num_points::Int64)
    @assert(num_points > 0)
    # As a guess, the most challenging integrals are the oscillatory
    # ones, so we'd like to have a quadrature suitable for them.
	h = pi/num_points
    points = (collect(range(0, pi, length=num_points+1))[1:end-1] + 
        collect(range(0, pi, length=num_points+1))[2:end]) ./ 2
	@assert(all(0 .<= points .<= pi))
    weights = h .* ones(num_points)
	@assert(length(weights) == length(points))
    for i = 1 : length(points)
        weights[i] = weights[i] * sin(points[i])
        points[i] = - cos(points[i])
    end
    @assert(all(weights .> 0), "Weights: "*string(weights))
    @assert(abs(sum(weights) - 2) < 0.02)
    fpoints = foil_points(a, points)
    vels = non_foil_ind_vel(a, fpoints)
    a.dw_positions = points
    a.dw_weights = weights
    a.dw_values = vels
    return
end

function add_particle_to_downwash_cache!(a::LAUTAT, ppos::Vector{Float32}, pvort::Real)
    @assert(length(ppos) == 2)
    @assert(length(a.dw_positions) > 0)
    fpoints = foil_points(a, a.dw_positions)
    vels = zeros(length(a.dw_positions), 2)
    for i = 1 : length(a.dw_positions)
        vels[i, :] = particle_induced_velocity(ppos, pvort, 
            fpoints[i, :], a.regularisation, a.reg_dist)
    end
    a.dw_values += vels
    return
end

function invalidate_downwash_cache!(a::LAUTAT)
    a.dw_values = zeros(0,0)
    a.dw_positions = []
    a.dw_weights = []
    return
end

function initialise(a::LAUTAT)
    tmptime = a.current_time
    a.current_time -= a.dt
    fill_downwash_cache!(a, 64)
    a.current_fourier_terms = compute_fourier_terms(a)
    a.last_fourier_terms = a.current_fourier_terms
    a.current_time = tmptime
    invalidate_downwash_cache!(a)
    return
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
    points, weights = FastGaussQuadrature.gausslegendre(64)
    vortex_pos = foil_points(a, points)
    weights .*= a.foil.semichord
    strengths = map(x->bound_vorticity_density(a, x), points).*weights
    kernel = a.regularisation
    vels = particle_induced_velocity(vortex_pos, strengths, mes_pnts, 
        kernel, a.reg_dist)    
    return vels
end

# Excludes U
function non_foil_ind_vel(a::LAUTAT, mes_pnts::Matrix{<:Real})  
    @assert(size(mes_pnts)[2] == 2, "size(mes_pnts)[2] should be 2, but is"*
        " actually "*string(size(mes_pnts)[2])*".")
    kernel = a.regularisation
    vels = particle_induced_velocity(a.te_particles.positions, 
        a.te_particles.vorts, mes_pnts, kernel, a.reg_dist)
    velsext = a.external_perturbation(mes_pnts, a.current_time)
    @assert(size(velsext) == size(mes_pnts), "A call to LAUTAT.external_perturbation("*
        "mes_pnts, time), with mes_pnts as a (N, 2) array should return an "*
        "(N, 2) array of velocities. Here, mes_pnts is "*string(size(mes_pnts))*
        " and returned velocities is "*string(size(velsext))*".")
    vels += velsext
    return vels
end

function te_wake_particle_velocities(a::LAUTAT)
    reg_dist = a.reg_dist
    kernel = a.regularisation
    vel_nf = non_foil_ind_vel(a, a.te_particles.positions)
    vel_foil = foil_induced_vel(a, a.te_particles.positions)
    vels = vel_nf + vel_foil
    vels[:, 1] .+= a.U[1]
    vels[:, 2] .+= a.U[2]
    return vels
end

function vel_normal_to_foil_surface(a::LAUTAT)
    mes_pnts = a.dw_positions
    @assert(all(-1 .<= mes_pnts .<= 1), "Foil in [-1,1]")
    field_vels = deepcopy(a.dw_values)
    ext_vels = a.U
    alpha = a.kinematics.AoA(a.current_time)
    alpha_dot = a.kinematics.dAoAdt(a.current_time)
    dzdt = a.kinematics.dzdt(a.current_time)
    slopes= map(a.foil.camber_slope, mes_pnts)
    rot = [cos(alpha) -sin(alpha); sin(alpha) cos(alpha)]
    for i = 1 : length(mes_pnts)
        field_vels[i, :] = rot * (field_vels[i, :] + ext_vels)
    end
    wash = (slopes .* (dzdt * sin(alpha) .+ field_vels[:, 1])
        .- alpha_dot * (a.foil.semichord .* mes_pnts .- a.kinematics.pivot_position)
        .+ dzdt * cos(alpha) .- field_vels[:, 2])
    return wash
end

function compute_fourier_terms(a::LAUTAT)
    points, weights = deepcopy(a.dw_positions), deepcopy(a.dw_weights)
	weights = weights ./ sqrt.(1 .- points.^2)
	points = acos.(.-points)
    dwsh = vel_normal_to_foil_surface(a)
    fterms = zeros(a.num_fourier_terms)
    for i = 1 : a.num_fourier_terms
        qpoints = cos.((i-1)*points) .* dwsh * 2 /
            (sqrt(a.U[1]^2 + a.U[2]^2) * pi)
        fterms[i] = sum(qpoints .* weights)
    end
    fterms[1] /= -2
    return fterms
end

function pivot_coordinate(foil::ThinFoilGeometry, kinem::RigidKinematics2D, t::Real)
    pos = [0., 0.]
    pos[1] = kinem.pivot_position
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
        vel .+= a.external_perturbation(part_pos, a.current_time)
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
    # Compute the influence of the known part of the wake
    ikpoints = a.dw_positions
    ikweights = a.dw_weights ./ sqrt.(1 .- ikpoints.^2)
    I_k = sum(
        (vel_normal_to_foil_surface(a) .* (-ikpoints .- 1) .* 
        2*a.foil.semichord ) .* ikweights)
    # And the bit that will be caused by the new particle
    posn = a.te_particles.positions[end,:]
    rot = [cos(alpha) -sin(alpha); sin(alpha) cos(alpha)]
    qpoints, qweights = FastGaussQuadrature.gausslegendre(64)
    qpoints, qweights = linear_remap(qpoints, qweights, -1, 1, 0, pi)
    function I_uk_integrand(theta::Vector{<:Real})
        foil_pos = -cos.(theta)
        foil_coords = foil_points(a, foil_pos)
        vels = mapreduce(
            i->(rot * particle_induced_velocity(posn, 1., foil_coords[i,:], a.regularisation, a.reg_dist))',
            vcat, 1 : length(qpoints))
        normal_vel = vels[:,1].*map(a.foil.camber_slope, foil_pos) .- vels[:,2]
        return normal_vel .* (cos.(theta).-1) * 2 * a.foil.semichord
    end
    I_uk = sum(I_uk_integrand(qpoints) .* qweights)
    # And now work out the vorticity
    vort = - (I_k + total_te_vorticity(a)) / (1 + I_uk)
    a.te_particles.vorts[end] = vort
    add_particle_to_downwash_cache!(a, posn, vort)
    return
end

function total_te_vorticity(a::LAUTAT)
    return sum(a.te_particles.vorts)
end

function leading_edge_suction_force(a::LAUTAT, density::Real)
    @assert(length(a.current_fourier_terms)==a.num_fourier_terms,
        "Fourier term vector length = "*string(length(a.current_fourier_terms))*
        " does not equal expected number of terms "*string(a.num_fourier_terms)
        *". Has this simulation been run yet?")
    return (a.U[1]^2 + a.U[2]^2) * pi * density * 2 * a.foil.semichord *
        a.current_fourier_terms[1]^2
end

function fourier_derivatives(a::LAUTAT)
    return (a.current_fourier_terms .- a.last_fourier_terms) ./ a.dt
end

function moment_coefficient(a::LAUTAT, xref::Real)
    Fn = aerofoil_normal_force(a, 1)
    AoA = a.kinematics.AoA(a.current_time)
    dAoAdt = a.kinematics.dAoAdt(a.current_time)
    hdot = a.kinematics.z_pos(a.current_time)
    fts = a.current_fourier_terms
    fds = fourier_derivatives(a)
    rot = [cos(AoA) -sin(AoA); sin(AoA) cos(AoA)]
    c = a.foil.semichord
    U = sqrt(a.U[1]^2 + a.U[2]^2)

    t1 = xref * Fn
    t21 = - pi * c^2 * U
    t2211 = (rot * a.U)[1] + hdot * sin(AoA)
    t2212 = (fts[1]/4 + fts[2]/4 - fts[3]/8)
    t221 = t2211 * t2212
    t222 = c * (7 * fds[1] / 16 + 11 * fds[2] / 64
        + fds[3] / 16 - fds[4] / 64)
    t22 = t221 + t222
    t2 = t21 * t22
    
    # Term 3 includes a weakly singular integral. We use singularity subtraction
    # to get round it.
    wake_ind_vel_ssm = -1 * 
        (rot * non_foil_ind_vel(a, foil_points(a, [-1]))')[1,1]
    points, weights = a.dw_positions, a.dw_weights
    normal_ind_vel_vect = a.dw_values
    normal_ind_velxr = zeros(size(normal_ind_vel_vect)[1])
    for i = 1 : size(normal_ind_vel_vect)[1]
        normal_ind_velxr[i] = points[i] * (rot * normal_ind_vel_vect[i, :])[1]
    end 
    tm1 = map(x->bound_vorticity_density(a, x), points)
    tm2 = (-1 .* normal_ind_velxr .- wake_ind_vel_ssm)
    t3 = a.foil.semichord * sum(weights .*
        tm1 .*
        tm2)
    t3 += wake_ind_vel_ssm * sqrt(a.U[1]^2 +  a.U[2]^2) * 2 *
        a.foil.semichord * pi * (a.current_fourier_terms[1] +
            a.current_fourier_terms[2]/2)

    t = (t1 + t2 + t3) / (U^2 * c^2 / 2)
    return t
end

function aerofoil_normal_force(a::LAUTAT, density::Real)
    AoA = a.kinematics.AoA(a.current_time)
    dAoAdt = a.kinematics.dAoAdt(a.current_time)
    hdot = a.kinematics.z_pos(a.current_time)
    fourier_derivs = fourier_derivatives(a)

    U_mag = sqrt(a.U[1]^2 + a.U[2]^2)
    rot = [cos(AoA) -sin(AoA); sin(AoA) cos(AoA)]
    t11 = density * pi * a.foil.semichord * 2 * U_mag
    t1211 = (rot * a.U)[1] + hdot * sin(AoA)
    t1212 = a.current_fourier_terms[1] + a.current_fourier_terms[2]/2
    t121 = t1211 * t1212
    t122 = 2 * a.foil.semichord * (
        (3/4) * fourier_derivs[1]
        + (1/4) * fourier_derivs[2] 
        + (1/8) * fourier_derivs[3])
    t12 = t121 + t122
    t1 = t11 * t12

    # Term 2 includes a weakly singular integral. We use singularity subtraction
    # to get round it.
    wake_ind_vel_ssm = (rot * non_foil_ind_vel(a, foil_points(a, [-1]))')[1,1]
    points, weights = a.dw_positions, a.dw_weights
    normal_ind_vel_vect = a.dw_values
    normal_ind_velxr = zeros(size(normal_ind_vel_vect)[1])
    for i = 1 : size(normal_ind_vel_vect)[1]
        normal_ind_velxr[i] = (rot * normal_ind_vel_vect[i, :])[1]
    end 
    tm1 = map(x->bound_vorticity_density(a, x), points)
    tm2 = (normal_ind_velxr .- wake_ind_vel_ssm)
    t2 = a.foil.semichord * density * sum(weights .*
        tm1 .* tm2)
    t2 += density * wake_ind_vel_ssm * sqrt(a.U[1]^2 +  a.U[2]^2) * 2 *
        a.foil.semichord * pi * (a.current_fourier_terms[1] +
            a.current_fourier_terms[2]/2)
    return t1 - t2
end

function lift_and_drag_coefficients(a::LAUTAT)
    schord = a.foil.semichord
    U = sqrt(a.U[1]^2 + a.U[2]^2)
    normal_coeff = aerofoil_normal_force(a, 1.) / (schord * U)
    suction_coeff = leading_edge_suction_force(a, 1.) / (schord * U)
    alpha = a.kinematics.AoA(a.current_time)
    cl = normal_coeff * cos(alpha) + suction_coeff * sin(alpha)
    cd = normal_coeff * sin(alpha) - suction_coeff * cos(alpha)
    return cl, cd
end

function to_vtk(a::LAUTAT, filename::String; include_foil=true)
    np = length(a.te_particles.vorts)
    extrap = include_foil ? 30 : 0
    extrac = include_foil ? extrap-1 : 0
    cells = Vector{WriteVTK.MeshCell}(undef, np + extrac)
    points = zeros(np+extrap, 3)
    vorts = zeros(np+extrap)
    vorts[1:np] = a.te_particles.vorts
    for i = 1 : np
        cells[i] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_VERTEX, [i])
        points[i, :] = [
            a.te_particles.positions[i,1], 
            a.te_particles.positions[i,2], 0.]
    end
    localpos = collect(-1:2/(include_foil ? extrac : 0):1)
    points[np+1:end, :] = hcat(foil_points(a, localpos), zeros(extrap, 1))
    bv = vcat([NaN], map(x->bound_vorticity_density(a, x), localpos[2:end]))
    vorts[np+1:end] = bv .* a.foil.semichord/extrap
    for i = 1 : extrac
        cells[i + np] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LINE, 
            [i + np, i + np + 1])
    end
    vtkfile = WriteVTK.vtk_grid(filename, points', cells)
    WriteVTK.vtk_point_data(vtkfile, vorts, "Vorticity")
    WriteVTK.vtk_save(vtkfile)
    return
end

function csv_titles(a::LAUTAT)
    return ["Time" "dt" "N" "BV" "z" "AoA" "A0" "A1" "Cl" "Cd"]
end

function csv_row(a::LAUTAT)
    cl, cd = lift_and_drag_coefficients(a)
    aoa = a.kinematics.AoA(a.current_time)
    z = a.kinematics.z_pos(a.current_time)
    return [a.current_time, a.dt, length(a.te_particles.vorts),
        bound_vorticity(a), z, aoa, a.current_fourier_terms[1],
        a.current_fourier_terms[2], cl, cd]'
end

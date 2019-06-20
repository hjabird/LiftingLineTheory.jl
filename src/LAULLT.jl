
"""
A vortex lattice with a "missing" edge when it is evaluated on i = 0.
Vorts are strengths[i, j]
Vertices are vertices[i, j, :] = [x_ij, y_ij, z_ij]
"""
mutable struct IncompleteVortexLattice
    vertices :: Array{Float64, 3}
    strengths :: Matrix{Float64}
    function IncompleteVortexLattice()
        return new(zeros(0,0,3),zeros(0,0))
    end
end

mutable struct LAULLT
    inner_sols :: LAUTAT
    inner_sol_positions :: Vector{Float64} # In (-1, 1)

    wing_platform :: StraightAnalyticWing
    wake_discretisation :: Union{Nothing, IncompleteVortexLattice}

    function LAULLT(;U=[1.,0]::Vector{<:Real}, 
        wing_platform=make_rectangular(StraightAnalyticWing, 4, 4),
        inner_solution_positions::Vector{<:Real}=collect(-1:2/8:1),
        foil::ThinFoilGeometry=ThinFoilGeometry(0.5,x->0),
        kinematics=RigidKinematics2D(x->x, x->0, 0.0),
        regularisation=winckelmans_regularisation(), reg_dist_factor=1.5,
        num_inner_fourier_terms=8,
        current_time=0.0, dt=0.025)

        @assert(all(-1 .< inner_sol_positions .< 1), "Inner solution span"*
            " positions must be in (-1, 1). Given "*string(inner_sol_positions))
        @assert(kinematics.pivot_positiong==0, "Kinematics pivot position "*
            "must be the midchord (0.0).")

        semispan = wing_platform.semispan
        semichord = map(wing_platform.chord_fn, semispan .* inner_sol_positions)
        foils = map(
            sc->ThinFoilGeometry(sc, foil.camber_line, foil.camber_slope),
            semichord)
        inner_probs = map(
            i->LAUTAT(;U=U, foil=foils[i], kinematics=kinematics,
                regularisation=regularisation, reg_dist_factor=reg_dist_factor,
                num_fourier_terms = num_inner_fourier_terms, 
                current_time=current_time, dt=dt),
            1:length(semichord))
        return new(inner_probs, inner_solution_positions, wing_platform, nothing)
    end
end

function induced_velocity(a::IncompleteVortexLattice, mes_pnts::Matrix{<:Real})
    @assert(size(mes_pnts)[2] == 3)
    nif = (size(a.vertices)[1]-1) * (size(a.vertices)[2])
    njf = (size(a.vertices)[1]-1) * (size(a.vertices)[2]-1)
    nf = nif + njf
    fstarts = zeros(nf, 3)
    fends = zeros(nf, 3)
    fstrs = zeros(nf)
    acc = 1
    function rstr(i, j)
        if (i > 0) && (j > 0) && (i < size(a.strengths)[1]) && (j < size(a.strengths)[2])
            ret = a.strengths[i, j]
        else
            ret = 0
        end
        return ret
    end
    for i = 1 : size(a.vertices)[1]-1
        for j = 1 : size(a.vertices)[2]
            fstarts[acc] = a.vertices[i, j, :]
            fends[acc] = a.vertices[i+1, j, :]
            fstrs[acc] = rstr(i, j) - rstr(i, j-1)
            acc += 1
        end
    end
    for i = 1 : size(a.vertices)[1]-1
        for j = 1 : size(a.vertices)[2]-1
            fstarts[acc] = a.vertices[i+1, j, :]
            fends[acc] = a.vertices[i+1, j+1, :]
            fstrs[acc] = rstr(i, j) - rstr(i+1, j)
            acc += 1
        end
    end
    vels = filament_induced_velocity(fstarts, fends, fstrs, mes_pnts)
    return vels
end

"""The 3D downwash on the lifting line collocation points."""
function outer_induced_downwash(a::LAULLT)
    span_positions = a.wing_planform.semispan * a.inner_sol_positions
    ni = length(span_positions)
    mes_pts = hcat(zeros(ni), span_positions, zeros(ni))
    outer_v = induced_velocity(a.wake_discretisation, mes_pts)
    common_v = induced_velocity(a, mes_pts)
    return outer_v - common_v
end

"""The translation that moves an inner 2D solution to the outer domain"""
function inner_to_outer_translation(a::LAULLT)
    ni = length(a.inner_sol_positions)
    trans = zeros(ni, 2)
    for i = 1:ni
        p = foil_position(a.inner_sols[i].foil, 1)
        trans[i, :] = -1 .* p
    end
    return trans
end

"""The 2D wake in the outer domain"""
function outer_2D_induced_downwash(a::LAULLT)
    kernel = singular_regularisation()
    ni = length(a.inner_sol_positions) #Numbef of Inner solutions
    dw = zeros(ni, 2) #DownWash
    translations = inner_to_outer_translation(a)
    for i = 1:ni
        opos = deepcopy(a.te_particles.positions) #Outer POSitions
        for j = 1:size(opos)[1]
            opos[j, :] += translations[i, :]
        end
        dw[i, :] = particle_induced_velocity(opos, a.te_particles.vorts,
            [0., 0.], kernel, 0.0)
    end
    return dw
end

"""Interpolate 2D wakes to create ring lattice in 3D & 2D points in outer 
domain"""
function construct_wake_lattice(a::LAULLT)
    np = length(a.inner_sols[1].te_particles.vorts) # Number of Particles
    ni = length(a.inner_sols) # Number of Inner solutoins
    semispan = a.wing_planform.semispan
    iypts = semispan * a.inner_solution_positions # Y PoinTS
    ypts = vcat([-semispan], (iypts[1:end-1] + iypts[2:end])/2, [semispan])
    vertices = zeros(np, length(ypts), 3)   # To make the mesh in the wake.
    vorticities = zeros(np, length(ypts)-1)
    vorticity_acc = map(i->-bound_vorticity(a.inner_sols[i]), 1:ni)
    # Now interpolate with cubic spline
    tmpvertex = zeros(ni, 2)
    # inner to outer translations
    translations = inner_to_outer_translation(a)
    for ix = 1:np
        for iy = 1:ni
            tmpvertex[iy, :] = a.inner_sols[iy].te_particles.positions[ix, :]
                + translations[iy]
        end

        spl_x = CubicSpline1D(iypts, tmpvertex[:, 1])
        spl_z = CubicSpline1D(iypts, tmpvertex[:, 2])
        for iy = 1:ni+1
            vertices[ix, iy, 1] = spl_x(ypts[iy])
            vertices[ix, iy, 2] = ypts[iy]
            vertices[ix, iy, 3] = spl_z(ypts[iy])
        end
        if ix < np
            for iy = 1 : ni
                vorticity_acc -= a.te_particles.vorts[ix]
                vorticities[ix+1, iy] = vorticity_acc[iy]
            end
        end
    end
    mesh = IncompleteVortexLattice(vertices, vorticities)
    return mesh
end

function apply_downwash_to_inner_solution!(a::LAULLT, downwashes::Matrix{<:Real})
    ni = length(a.inner_sols)
    for i = 1:ni
        a.inner_sols[i].external_perturbation = (x,t)->downwashes[i, :]
    end
    return
end

function advance_one_step(a::LAULLT)
    if typeof(a.wake_discretisation) == Nothing
        a.wake_discretisation = construct_wake_lattice(a)
    end
    outer_downwashes = outer_induced_downwash(a)
    apply_downwash_to_inner_solution!(a, outer_downwashes)
    for i = 1 : length(a.inner_sols)
        advance_one_step(a.inner_sols[i])
    end
    a.wake_discretisation = construct_wake_lattice(a)
    return
end

function to_vtk(a::LAULLT)
    
    nif = (size(a.vertices)[1]-1) * (size(a.vertices)[2])
    njf = (size(a.vertices)[1]-1) * (size(a.vertices)[2]-1)
    points = reshape(a.wake_discretisation, length(a.wake_discretisation.vorts), 3)
    cells = Vector{WriteVTK.MeshCell}(undef, size)
end

function lift_coefficient(a::LAULLT)
    @error("TODO")
    return
end

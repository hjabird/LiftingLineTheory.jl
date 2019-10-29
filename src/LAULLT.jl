#
# LAULLT.jl
#
# Large amplitude unsteady lifting line theory. A lifting line theory based
# upon unsteady thin aerofoil theory.
#
# Copyright HJAB 2019
#
################################################################################

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
    function IncompleteVortexLattice(
        vertices::Array{<:Real, 3},
        ring_strengths::Matrix{<:Real})
        verticies_size = size(vertices)
        strength_size = size(ring_strengths)
        @assert((verticies_size[1]-1 == strength_size[1]) || 
            ((strength_size[1]==0) && (verticies_size[1]==0)),"Dimension 1 of "*
            "vertices array must be 1 larger than dimension 1 of strength "*
            "matrix or both 0. Dimensions are of size "
            *string(verticies_size[1])*" and "*string(strength_size[1])*".")
        @assert(verticies_size[2]-1 == strength_size[2] ||
            ((strength_size[2]==0) && (verticies_size[2]==0)),"Dimension 2 of "*
            "vertices array must be 1 larger than dimension 2 of strength "*
            "matrix or both 0. Dimensions are of size "
            *string(verticies_size[2])*" and "*string(verticies_size[2])*
            " and "*string(strength_size[2])*".")
        @assert(verticies_size[3]==3, "size(vertices_array)[3] must == 3. Is "*
            string(verticies_size[3])*".")
        return new(vertices, ring_strengths)
    end
end

mutable struct LAULLT
    inner_sols :: Vector{LAUTAT}
    inner_sol_positions :: Vector{Float64} # In (-1, 1)

    wing_planform :: StraightAnalyticWing
    wake_discretisation :: Union{Nothing, IncompleteVortexLattice}
    segmentation :: Vector{Float64}

    function LAULLT(;U=[1.,0]::Vector{<:Real}, 
        wing_planform=make_rectangular(StraightAnalyticWing, 4, 4),
        inner_solution_positions::Vector{<:Real}=
            (collect(-1:2/16:1)[1:end-1] + collect(-1:2/16:1)[2:end]) ./ 2,
        foil::ThinFoilGeometry=ThinFoilGeometry(0.5,x->0),
        kinematics=RigidKinematics2D(x->x, x->0, 0.0),
        regularisation=winckelmans_regularisation(), reg_dist=NaN,
        num_inner_fourier_terms=8,
        current_time=0.0, dt=0.025, segmentation=[])

        @assert(all(-1 .< inner_solution_positions .< 1), "Inner solution span"*
            " positions must be in (-1, 1). Given "*
            string(inner_solution_positions))
        @assert(length(U)==2, "U must be a vector of length two: [u, w]")
        @assert(isnan(reg_dist) ? true : reg_dist>=0, 
            "Regularisation distance must be positive.")

        semispan = wing_planform.semispan
        semichord = map(x->wing_planform.chord_fn(x)/2, semispan .* 
            inner_solution_positions)
        if isnan(reg_dist)
            reg_dist = dt * 1.5 * sqrt(U[1]^2 + U[2]^2)
        end
        if length(segmentation) == 0
            segmentation = vcat(-1: 2 / (length(inner_solution_positions)*4 + 1): 1)
        else
            @assert(segmentation[end] == 1, "Segmentation must be in [-1, 1]")
            @assert(segmentation[1] == -1, "Segmentation must be in [-1, 1]")
            if length(segmentation)-1 < length(inner_solution_positions)
                @warn("Segmentation for outer solution is less refined than"*
                    " inner solution. IE length(segmenation)-1 < "*
                    "length(inner_solution_positions. Lengths are "*
                    string(length(segmentation))*" and "*
                    string(length(inner_solution_positions))*"respectively.")
            end
        end
        foils = map(
            sc->ThinFoilGeometry(sc, foil.camber_line, foil.camber_slope),
            semichord)
        inner_probs = map(
            i->LAUTAT(;U=U, foil=foils[i], kinematics=kinematics,
                regularisation=regularisation, reg_dist=reg_dist,
                num_fourier_terms = num_inner_fourier_terms, 
                current_time=current_time, dt=dt),
            1:length(semichord))
        for i = 1 : length(inner_probs)
            initialise!(inner_probs[i])
        end
        return new(inner_probs, inner_solution_positions, wing_planform, nothing,
            segmentation)
    end
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

#=
The velocity induced by an incomplete vortex lattice.
For the first i row (i=1) the filaments are ignored.
=#
function induced_velocity(a::IncompleteVortexLattice, mes_pnts::Matrix{<:Real})
    @assert(size(mes_pnts)[2] == 3)
    if all(size(a.vertices)[1:2] .>= 2)
        fstarts, fends, fstrs = to_filaments(a)
        vels = filament_induced_velocity(fstarts, fends, fstrs, mes_pnts)
    else
        # The Lattice conatains no rings.
        @assert(prod(size(a.strengths)) == 0)
        vels = zeros(Float32, size(mes_pnts)[1], 3)
    end
    return vels
end

"""The 3D downwash on the lifting line collocation points."""
function outer_induced_downwash(a::LAULLT)
    span_positions = a.wing_planform.semispan * a.inner_sol_positions
    ni = length(span_positions)
    mes_pts = hcat(zeros(ni), span_positions, zeros(ni))
    outer_v = induced_velocity(a.wake_discretisation, mes_pts)
    common_v = outer_2D_induced_downwash(a)
    @assert(size(outer_v)[1] == size(common_v)[1])
    for i = 1 : size(outer_v)[1]
        outer_v[i, :] -= [common_v[i,1], 0, common_v[i, 2]]
    end
    return outer_v
end

"""The translation that moves an inner 2D solution to the outer domain"""
function inner_to_outer_translation(a::LAULLT)
    ni = length(a.inner_sol_positions)
    trans = zeros(ni, 2)
    for i = 1:ni
        p = foil_points(a.inner_sols[i], [1])
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
        opos = deepcopy(a.inner_sols[i].te_particles.positions) #Outer POSitions
        for j = 1:size(opos)[1]
            opos[j, :] += translations[i, :]
        end
        dw[i, :] = particle_induced_velocity(opos, 
           a.inner_sols[i].te_particles.vorts, [0., 0.], kernel, 0.0)
    end
    return dw
end

"""Interpolate 2D wakes to create ring lattice in 3D & 2D points in outer 
domain"""
function construct_wake_lattice(a::LAULLT)
    np = length(a.inner_sols[1].te_particles.vorts) # Number of Particles
    ni = length(a.inner_sols) # Number of Inner solutoins
    semispan = a.wing_planform.semispan
    iypts = semispan * a.inner_sol_positions    # inner y solution posn.
    ypts = semispan .* a.segmentation           # interpolation positions
    cypts = (ypts[1:end-1] + ypts[2:end]) ./ 2  # Between interpolation positions.
    vertices = zeros(np+1, length(ypts), 3)     # To make the mesh in the wake.
    vorticities = zeros(np, length(ypts)-1)     # Vortex ring strengths.
    vorticity_acc = map(i->-bound_vorticity(a.inner_sols[i]), 1:ni)
    # One edge of the vortex lattice is the lifting line on x=0,z=0
    for iy = 1 : length(ypts)
        vertices[1, iy, :] = [0, ypts[iy], 0]
    end
    if np > 0
        #spl_v = CubicSpline{Float64}(vcat([-semispan], iypts, [semispan]), vcat([0], vorticity_acc, [0]))
        spl_v = CubicSpline{Float64}(iypts, vorticity_acc)
        for iy = 1 : length(cypts)
            vorticities[1, iy] = spl_v(cypts[iy])
        end
    end
    # Now rings off the lifting line:
    tmpvertex = zeros(ni, 2)    # Locations of particles in outer solution.
    # inner to outer translations
    translations = inner_to_outer_translation(a)
    #   ix: particle index. ix = 1 is first shed particle. ix = np is most
    #       recently shed particle.
    #   iy: spanwise index. 1:ni is for actual inner solutions. 1:length(cypts)
    #       is for interpolated values.
    for ix = 1:np
        for iy = 1:ni
            tmpvertex[iy, :] = a.inner_sols[iy].te_particles.positions[np-ix+1, :] + 
                translations[iy, :]
        end
        spl_x = CubicSpline{Float64}(iypts, tmpvertex[:, 1])
        spl_z = CubicSpline{Float64}(iypts, tmpvertex[:, 2])
        for iy = 1:length(ypts)
            vertices[ix + 1, iy, 1] = spl_x(ypts[iy])
            vertices[ix + 1, iy, 2] = ypts[iy]
            vertices[ix + 1, iy, 3] = spl_z(ypts[iy])
        end
        if ix < np  # The final vortex particle is implicitly correctly set.
            vorticity_acc -= map(i->a.inner_sols[i].te_particles.vorts[np-ix+1], 1:ni)
            # spl_v = CubicSpline{Float64}(vcat([-semispan], iypts, [semispan]), vcat([0], vorticity_acc, [0]))
            spl_v = CubicSpline{Float64}(iypts, vorticity_acc)
            for iy = 1 : length(cypts)
                vorticities[ix+1, iy] = spl_v(cypts[iy])
            end
        end
    end
    @assert(all(isfinite.(vorticities)), "Not all vortex rings had finite strengths.")
    @assert(all(isfinite.(vertices)), "Not all vertices had finite coordinates.")
    mesh = IncompleteVortexLattice(vertices, vorticities)
    return mesh
end

function to_filaments(a::IncompleteVortexLattice)
    nif = (size(a.vertices)[1]-1) * (size(a.vertices)[2])   # i-dir (streamwise)
    njf = (size(a.vertices)[1]-1) * (size(a.vertices)[2]-1) # j-dir (spanwise)
    nf = nif + njf
    fstarts = zeros(nf, 3)
    fends = zeros(nf, 3)
    fstrs = zeros(nf)
    acc = 1
    function rstr(i, j)
        @assert(i > 0, "You're not including i=1!")
        if (i > 0) && (j > 0) && (i <= size(a.strengths)[1]) && (j <= size(a.strengths)[2])
            ret = a.strengths[i, j]
        else
            ret = 0
        end
        return ret
    end
    # Streamwise filaments. (i --- i+1)
    for i = 1 : size(a.vertices)[1]-1
        for j = 1 : size(a.vertices)[2]
            fstarts[acc, :] = a.vertices[i, j, :]
            fends[acc, :] = a.vertices[i+1, j, :]
            fstrs[acc] = rstr(i, j) - rstr(i, j-1)
            acc += 1
        end
    end
    @assert(acc - 1 == nif)
    # Spanwise filaments. (j --- j+1)
    for i = 2 : size(a.vertices)[1]
        for j = 1 : size(a.vertices)[2]-1
            fstarts[acc, :] = a.vertices[i, j, :]
            fends[acc, :] = a.vertices[i, j+1, :]
            fstrs[acc] = rstr(i-1, j) - rstr(i, j)
            acc += 1
        end
    end
    @assert(acc - 1 == nif + njf)
    return fstarts, fends, fstrs
end

function apply_downwash_to_inner_solution!(a::LAULLT, downwashes::Matrix{<:Real})
    @assert(size(downwashes)[1] == length(a.inner_sols),
        "There should be 1 downwash per inner solution, but instead there are"*
        string(size(downwashes)[1])*" downwashes and "*
        string(length(a.inner_sols))*" inner solutions.")
    @assert(size(downwashes)[2]==3, "Expected downwashes to have size (N, 3).")
    @assert(all(isfinite.(downwashes)), "Not all downwash values are finite.")
    ni = length(a.inner_sols)
    dw2d = hcat(downwashes[:, 1], downwashes[:, 3])

    function mk_uniform_field(x, dwuinform)
        vels = zeros(size(x)[1], 2)
        for i = 1 : size(x)[1]
            vels[i, :] += dwuinform
        end
        return vels
    end

    for i = 1:ni
        a.inner_sols[i].external_perturbation = 
            (x,t)->mk_uniform_field(x, dw2d[i, :])
    end
    return
end

# Returns local C_l * chord and C_d * chord as spline.
function lift_and_drag_coefficient_splines(a::LAULLT)
    semispan = a.wing_planform.semispan
    ypts = semispan * a.inner_sol_positions
    eypts = vcat([-semispan], ypts, [semispan])
    lifts = zeros(length(eypts))
    drags = zeros(length(eypts))
    lifts[1] = 0
    drags[1] = 0
    for i = 1 : length(eypts)-2
        lifts[i+1], drags[i+1] = lift_and_drag_coefficients(a.inner_sols[i])
        lifts[i+1] *= chord(a.wing_planform, a.inner_sol_positions[i])
        drags[i+1] *= chord(a.wing_planform, a.inner_sol_positions[i])
    end
    lifts[end] = 0
    drags[end] = 0
    ls = CubicSpline{Float64}(eypts, lifts)
    ds = CubicSpline{Float64}(eypts, drags)
    return ls, ds
end

# 2D C_l at a spanwise position (normalised wrt/ chord)
function lift_and_drag_coefficients(a::LAULLT, y::Real)
    semispan = a.wing_planform.semispan
    @assert(-semispan <= y <= semispan)
    ls, ds =  lift_and_drag_coefficient_splines(a)
    c = chord(a.wing_planform, y)
    return ls(y) / c, ds(y) / c
end

function lift_and_drag_coefficients(a::LAULLT)
    semispan = a.wing_planform.semispan
    warea = area(a.wing_planform)
    ls, ds = lift_and_drag_coefficient_splines(a)
    points, weights = FastGaussQuadrature.gausslegendre(50)
    points, weights = linear_remap(points, weights, -1, 1, -semispan, semispan)
    chords = map(x->chord(a.wing_planform, x), points)
    liftc = sum(weights .* ls.(points)) / warea
    dragc = sum(weights .* ds.(points)) / warea
    return liftc, dragc
end

function lift_coefficient(a::LAULLT, y::Real)
    liftc, ~ = lift_and_drag_coefficients(a, y)
    return liftc
end

function lift_coefficient(a::LAULLT)
    liftc, ~ = lift_and_drag_coefficients(a)
    return liftc
end

function drag_coefficient(a::LAULLT, y::Real)
    ~, dragc = lift_and_drag_coefficients(a, y)
    return dragc
end

function drag_coefficient(a::LAULLT)
    ~, dragc = lift_and_drag_coefficients(a)
    return dragc
end

function to_vtk(a::LAULLT, filename::String)
    wake = a.wake_discretisation
    if prod(size(wake.vertices))!= 0
        fstarts, fends, fstrs = to_filaments(wake)
        nfils = size(fstarts)[1]
        points = vcat(fstarts, fends)
        cells = Vector{WriteVTK.MeshCell}(undef, nfils)        
        celldata = fstrs
        for i = 1 : nfils
            cells[i] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LINE, 
                [i, i+nfils])
        end
    else
        points = zeros(0, 3)
        cells = Vector{WriteVTK.MeshCell}(undef, 0)
        cell_str = zeros(0)
    end
    vtkfile = WriteVTK.vtk_grid(filename, points', cells)
    WriteVTK.vtk_cell_data(vtkfile, celldata, "Vorticity")
    WriteVTK.vtk_save(vtkfile)
    return
end

function csv_titles(a::LAULLT)
    start = ["Time" "dt" "N" "N_Inner" "CL" "CD"]
    for i = 1 : length(a.inner_sols)
        str = "I"*string(i)*"_"
        tmp = [str*"A0" str*"A1" str*"U3Dx" str*"U3Dz"]
        start = hcat(start, tmp)
    end
    return start
end

function csv_row(a::LAULLT)
    inner1 = a.inner_sols[1]
    ctime = inner1.current_time
    lift, drag = lift_and_drag_coefficients(a)
    start = [ctime inner1.dt length(inner1.te_particles.vorts) length(a.inner_sols) lift drag]
    for i = 1 : length(a.inner_sols)
        inner = a.inner_sols[i]
        tmp = [inner.current_fourier_terms[1] inner.current_fourier_terms[2] inner.external_perturbation([0 0], ctime)[1,1] inner.external_perturbation([0 0], ctime)[1,2]]
        start = hcat(start, tmp)
    end
    return start
end

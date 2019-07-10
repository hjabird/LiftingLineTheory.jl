
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

    function LAULLT(;U=[1.,0]::Vector{<:Real}, 
        wing_planform=make_rectangular(StraightAnalyticWing, 4, 4),
        inner_solution_positions::Vector{<:Real}=collect(-1:2/12:1)[2:end-1],
        foil::ThinFoilGeometry=ThinFoilGeometry(0.5,x->0),
        kinematics=RigidKinematics2D(x->x, x->0, 0.0),
        regularisation=winckelmans_regularisation(), reg_dist_factor=1.5,
        num_inner_fourier_terms=8,
        current_time=0.0, dt=0.025)

        @assert(all(-1 .< inner_solution_positions .< 1), "Inner solution span"*
            " positions must be in (-1, 1). Given "*
            string(inner_solution_positions))
        @assert(kinematics.pivot_position==0, "Kinematics pivot position "*
            "must be the midchord (0.0).")

        semispan = wing_planform.semispan
        semichord = map(wing_planform.chord_fn, semispan .* 
            inner_solution_positions)
        foils = map(
            sc->ThinFoilGeometry(sc, foil.camber_line, foil.camber_slope),
            semichord)
        inner_probs = map(
            i->LAUTAT(;U=U, foil=foils[i], kinematics=kinematics,
                regularisation=regularisation, reg_dist_factor=reg_dist_factor,
                num_fourier_terms = num_inner_fourier_terms, 
                current_time=current_time, dt=dt),
            1:length(semichord))
        for i = 1 : length(inner_probs)
            initialise(inner_probs[i])
        end
        return new(inner_probs, inner_solution_positions, wing_planform, nothing)
    end
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
            fstrs[acc] = rstr(i, j) - rstr(i+1, j)
            acc += 1
        end
    end
    @assert(acc - 1 == nif + njf)
    return fstarts, fends, fstrs
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
    iypts = semispan * a.inner_sol_positions # inner y solution posn.
    #ypts = vcat([-semispan], (iypts[1:end-1] + iypts[2:end])/2, [semispan])
    ypts = collect(-semispan: semispan/20 : semispan)
    cypts = (ypts[1:end-1] + ypts[2:end]) ./ 2
    #@assert(length(ypts) == ni+1)
    vertices = zeros(np+1, length(ypts), 3)   # To make the mesh in the wake.
    vorticities = zeros(np, length(ypts)-1)
    vorticity_acc = map(i->-bound_vorticity(a.inner_sols[i]), 1:ni)
    # One edge of the vortex lattice is the lifting line on x=0,z=0
    for iy = 1 : length(ypts)
        vertices[1, iy, :] = [0, ypts[iy], 0]
    end
    tmpvertex = zeros(ni, 2)
    # inner to outer translations
    translations = inner_to_outer_translation(a)
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
        if ix < np
            vorticity_acc -= map(i->a.inner_sols[i].te_particles.vorts[ix], 1:ni)
            spl_v = CubicSpline{Float64}(iypts, vorticity_acc)
            for iy = 1 : length(cypts)
                vorticities[ix+1, iy] = spl_v(cypts[iy])
            end
        end
    end
    mesh = IncompleteVortexLattice(vertices, vorticities)
    return mesh
end

function apply_downwash_to_inner_solution!(a::LAULLT, downwashes::Matrix{<:Real})
    @assert(size(downwashes)[1] == length(a.inner_sols),
        "There should be 1 downwash per inner solution, but instead there are"*
        string(size(downwashes)[1])*" downwashes and "*
        string(length(a.inner_sols))*" inner solutions.")
    @assert(size(downwashes)[2]==3, "Expected downwashes to have size (N, 3).")
    ni = length(a.inner_sols)
    dw2d = hcat(downwashes[:, 1], downwashes[:, 2])

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

function lift_coefficient(a::LAULLT)
    @error("TODO")
    return
end

function csv_titles(a::LAULLT)
    start = ["Time" "dt" "N" "N_Inner"]
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
    start = [ctime inner1.dt length(inner1.te_particles.vorts) length(a.inner_sols)]
    for i = 1 : length(a.inner_sols)
        inner = a.inner_sols[i]
        tmp = [inner.current_fourier_terms[1] inner.current_fourier_terms[2] inner.external_perturbation([0 0], ctime)[1,1] inner.external_perturbation([0 0], ctime)[1,2]]
        start = hcat(start, tmp)
    end
    return start
end

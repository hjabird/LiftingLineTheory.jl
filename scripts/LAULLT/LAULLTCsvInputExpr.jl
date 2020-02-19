#=
A comparison of LAULLT and Sclavounos' theory for small amplitude
heave of an AR3 wing.
=#

using LiftingLineTheory
using PyPlot
using DelimitedFiles

let
    AR = 6
    wing = LiftingLineTheory.make_rectangular(StraightAnalyticWing, AR, AR)
    dt = 0.025
    ninner = 48
    kinem_file = "nonsmooth_fast_pitch_eld_3deg_0.dat"
    casename = "AR"*string(AR)*"Rect_LEpNonsmoothFast3_Ni"*string(ninner)*"_dt"*string(dt)
    casename = replace(casename, "." => "p")

	println("Case "*casename)
	println("AR = ", AR)
    println("dt = ", dt)
    println("dtstar = ", 1 / wing.chord_fn(0) * dt)

    bigsegs= range(-1, 1, length=ninner+1)
    innersolpos = collect((bigsegs[2:end]+bigsegs[1:end-1])./2)
    segs = collect(range(-1, 1, length=1*ninner+1))

    function read_openfoam_kinematics(file_path)
        local file = open(file_path, "r")
        local line = readline(file)
        local n = parse(Int, line)    # Number of entries
        local kinem_mat_3d = zeros(n, 7)
        readline(file)          # "(" line 
        for i = 1:n
            line = readline(file)
            line = replace(line, '('=>' ')
            line = replace(line, ')'=>' ')
            local values = split(line)
            @assert(length(values)==7)
            kinem_mat_3d[i, :] = map(v->parse(Float64, v), values)
        end
        close(file_path)
        return kinem_mat_3d
    end
    function openfoam_kinem_to_rigidkinem(kinem_mat_3d, pivot_pos, scale)
        local k2d = zeros(size(kinem_mat_3d)[1], 6)
        k2d[:, 1] =  kinem_mat_3d[:, 1]
        k2d[:, 2] =  kinem_mat_3d[:, 3] .* scale
        k2d[:, 4] =  deg2rad.(kinem_mat_3d[:, 7]) .* scale
        k2d[:, 6] .=  pivot_pos
        k2d[1:end-1, 3] = (k2d[2:end, 2]-k2d[1:end-1, 2]) ./ (k2d[2:end, 1]-k2d[1:end-1, 1])
        k2d[end, 4] = k2d[end-1, 4]
        k2d[1:end-1, 5] = (k2d[2:end, 4]-k2d[1:end-1, 4]) ./ (k2d[2:end, 1]-k2d[1:end-1, 1])
        k2d[end, 5] = k2d[end-1, 5]
        local starttime = minimum(k2d[:,1])
        local endtime = maximum(k2d[:,1])
        return LiftingLineTheory.from_matrix(RigidKinematics2D, k2d), starttime, endtime
    end


    kinem, stime, etime = openfoam_kinem_to_rigidkinem(read_openfoam_kinematics(kinem_file), -0.5, 1) # CHANGE ME!!!!
    nsteps = Int64(floor((etime-stime) / dt))
    println("nsteps = ", nsteps)

    prob = LAULLT(;kinematics=kinem,
        wing_planform=wing, dt=dt, segmentation=segs, inner_solution_positions=innersolpos,
        current_time=stime, )
    println("n_inner = ", length(prob.inner_sols))
    println("inner_sol_pos = ", prob.inner_sol_positions)
    println("Segmentation = ", prob.segmentation)
    
    hdr = csv_titles(prob)
    rows = zeros(0, length(hdr))
    print("\nLAULLT\n")
    for i = 1 : nsteps
        print("\rStep ", i, " of ", nsteps, ".\t\t\t\t\t")
        advance_one_step(prob)
        rows = vcat(rows, csv_row(prob))
    end
    to_vtk(prob, casename)

    writefile = open(casename*".csv", "w")
    writedlm(writefile, hdr, ", ")
    writedlm(writefile, rows, ", ")
    close(writefile)

    #return prob, rows, hdr
end

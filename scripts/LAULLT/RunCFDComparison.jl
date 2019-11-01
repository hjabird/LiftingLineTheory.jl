#=
Compare LAULLT to CFD.
Optional addition of LAUTAT to CFD
=#

using LiftingLineTheory
using PyPlot
using DelimitedFiles
using CVortex

function run_laullt_cfd_comp(
    wing :: StraightAnalyticWing,
    kinem :: RigidKinematics2D,
    cfd_res_path :: String;
    run_2d::Bool=true,
    cfd_res_path2d::String="",
    tmin = 0,
    tmax = -1,
    cfd_res_toff::Real=0,
    cfd_res2d_toff::Real=0
)
    cfd_results = readdlm(cfd_res_path,'\t';header=true) 
    cfd_rows = cfd_results[1]

    if tmax < 0
        tmax = maximum(cfd_rows[:,1]) - cfd_res_toff
    end

    dt = 0.015
    prob = LAULLT(wing, kinem; num_inner_solutions=32, dt=dt)

    hdr = csv_titles(prob)
    rows = zeros(0, length(hdr))

    nsteps = Int64(ceil((tmax-tmin)/dt))
    for i = 1 : nsteps
        print("\rStep ", i, " of ", nsteps, ".\t\t\t\t\t")
        advance_one_step(prob)
        #to_vtk(prob, "test_"*string(i))
        rows = vcat(rows, csv_row(prob))
    end

    # 2D problem using LAUTAT
    if run_2d
        prob2d = LAUTAT(;kinematics=kinem, dt=dt, regularisation=singular_regularisation())
        nsteps = Int64(ceil((tmax-tmin)/dt))
        hdr2d = csv_titles(prob2d)
        rows2d = zeros(0, length(hdr2d))
        for i = 1 : nsteps
            print("\rLAUTAT step ", i, " of ", nsteps, ".\t\t\t\t\t")
            advance_one_step(prob2d)
            #to_vtk(prob, "test_"*string(i))
            rows2d = vcat(rows2d, csv_row(prob2d))
        end
    end


    if cfd_res_path2d != ""
        cfd_results2d = readdlm(cfd_res_path2d,'\t';header=true) 
        cfd_rows2d = cfd_results2d[1]
    end

    figure()
    plot(rows[:, 1], rows[:, 5], label="LAULLT (3D)")
    if run_2d
        println("plotting 2D")
        plot(rows2d[:, 1], rows2d[:, 9], label="LAUTAT (2D)")
    end
    plot(cfd_rows[:, 1] .- cfd_res_toff, cfd_rows[:, 2], label="CFD (3D)")    
    if cfd_res_path2d != ""
        println("plotting 2D CFD")
        plot(cfd_rows2d[:, 1] .- cfd_res2d_toff, cfd_rows2d[:, 2], label="CFD (2D)")
    end
    xlabel("t (s)")
    ylabel("C_L")
    legend()

    return prob, rows, hdr
end

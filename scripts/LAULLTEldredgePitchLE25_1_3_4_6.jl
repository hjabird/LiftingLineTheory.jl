#=
Use LAULLT to simulate a large amplitude pitching problem
=#

using LiftingLineTheory
using PyPlot
using DelimitedFiles

let
    AR = 3
    wing = LiftingLineTheory.make_rectangular(StraightAnalyticWing, AR, AR)
    dt = 0.015

    t0 = 0
    t1 = 1
    t2 = 3
    t3 = 4
    t4 = 6
    tmax = 7
    amp = deg2rad(25)
    a=11
    U_inf = 1.
    chord = 1
	
	println("Comparing LAULLT CFD in pitch.")
	println("Ampitude is 25 degrees. t1=1s, t2=3s, t3=4s, t4=6s, tmax=7s")

    wing = make_rectangular(StraightAnalyticWing, 4, 4)
    eld_fn = t->eldredge_ramp(t, t1, t2, t3, t4, a, U_inf, chord)
    eld_max = eld_fn((t2+t3)/2)
    kinematics = make_pitch_function(RigidKinematics2D, -0.5, t->amp * eld_fn(t) / eld_max)
    prob = LAULLT(wing, kinematics; num_inner_solutions=32, dt=dt)

    hdr = csv_titles(prob)
    rows = zeros(0, length(hdr))

    nsteps = Int64(ceil((tmax-t0)/dt))
    for i = 1 : nsteps
        print("\rStep ", i, " of ", nsteps, ".\t\t\t\t\t")
        advance_one_step(prob)
        #to_vtk(prob, "test_"*string(i))
        rows = vcat(rows, csv_row(prob))
    end

    # 2D problem using LAUTAT
    prob2d = LAUTAT(;kinematics=kinematics, dt=dt, regularisation=singular_regularisation())
    nsteps = Int64(ceil((tmax-t0)/dt))
    hdr2d = csv_titles(prob2d)
    rows2d = zeros(0, length(hdr2d))
    for i = 1 : nsteps
        print("\rLAUTAT step ", i, " of ", nsteps, ".\t\t\t\t\t")
        advance_one_step(prob2d)
        #to_vtk(prob, "test_"*string(i))
        rows2d = vcat(rows2d, csv_row(prob2d))
    end

    cfd_results = readdlm("scripts/CFD_results_ref/eldredge_pitchLE25_Re10k_1_3_4_6.dat",',';header=true) 
    cfd_rows = cfd_results[1]

    figure()
    plot(rows[:, 1], rows[:, 5], label="LAULLT")
    plot(rows2d[:, 1], rows2d[:, 9], label="LAUTAT")
    plot(cfd_rows[:, 1] .- 250, cfd_rows[:, 4], label="CFD")
    xlabel("t (s)")
    ylabel("C_L")
    legend()
#=
    figure()
    plot(rows[:, 1], rows[:, 5], label="LAULLT")
    plot(cfd_rows[:, 1] .- 250, cfd_rows[:, 4], label="CFD")
    xlabel("t (s)")
    ylabel("C_M")
    legend()
=#
    return prob, rows, hdr
end

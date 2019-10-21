#=
A comparison of LAULLT and Wagner/
=#

using LiftingLineTheory
using PyPlot

let
    AR = 8
    wing = LiftingLineTheory.make_rectangular(StraightAnalyticWing, AR, AR)
    amp = deg2rad(2)
    dt = 0.015
    nsteps = 10

	
	println("Comparing LAULLT to Wagner.")
	println("AR = ", AR)
	println("dt = ", dt)
    println("amp = ", amp)
    
    ninner = 8
    bigsegs= range(-1, 1, length=ninner+1)
    innersolpos = collect((bigsegs[2:end]+bigsegs[1:end-1])./2)
    segs = collect(range(-1, 1, length=1*ninner+1))


    #prob = LAULLT(;kinematics=RigidKinematics2D(x->amp*cos(omega*x), x->0, 0.5),
    #    wing_planform=wing, dt=dt)
    #prob = LAULLT(;kinematics=RigidKinematics2D(x->0, x->deg2rad(5), 0),
    #    wing_planform=wing, dt=dt, segmentation=segmentation)
    prob = LAULLT(;kinematics=RigidKinematics2D(x->0, x->amp, 0.0),
        wing_planform=wing, dt=dt, segmentation=segs, inner_solution_positions=innersolpos)
    println("n_inner = ", length(prob.inner_sols))
    println("inner_sol_pos = ", prob.inner_sol_positions)
    println("Segmentation = ", prob.segmentation)
    
    hdr = csv_titles(prob)
    rows = zeros(0, length(hdr))
    print("\nLAULLT\n")
    for i = 1 : nsteps
        print("\rStep ", i, " of ", nsteps, ".\t\t\t\t\t")
        advance_one_step(prob)
        to_vtk(prob, "test_"*string(i))
        rows = vcat(rows, csv_row(prob))
    end
    plot(rows[50:end, 1], rows[50:end, 5], label="LAULLT")

    prob2d = LAUTAT(;kinematics=RigidKinematics2D(x->0, x->amp, 0.0), dt=dt)
    hdr2d = csv_titles(prob2d)
    rows2d = zeros(0, length(hdr2d))
    print("\nLAUTAT\n")
    for i = 1 : nsteps
        print("\rStep ", i, " of ", nsteps, ".\t\t\t\t\t")
        advance_one_step(prob2d)
        rows2d = vcat(rows2d, csv_row(prob2d))
    end
    println("\n")
    plot(rows2d[50:end, 1], rows2d[50:end, 9], label="LAUTAT")

    xlabel("Time")
    ylabel("C_L")
    legend()

    figure()
    plot(rows[50:end, 1], rows[50:end, 7], label="LAULLT A0 edge")
    plot(rows[50:end, 1], rows[50:end, 7+4*4], label="LAULLT A0 5")
    plot(rows2d[50:end, 1], rows2d[50:end, 4], label="LAUTAT A0")
    plot(rows[50:end, 1], rows[50:end, 8], label="LAULLT A1 edge")
    plot(rows[50:end, 1], rows[50:end, 8+4*4], label="LAULLT A1 5")
    plot(rows2d[50:end, 1], rows2d[50:end, 8], label="LAUTAT A1")
    legend()

    println(hdr)
    println(hdr2d)
    figure()
    for i = 1 : Int(ninner/2)
        plot(rows[50:end, 1], rows[50:end, 6 + i * 4], label="LAULLT Uz "*string(i))
    end
    legend()
    return prob, rows, hdr
end

#=
A comparison of LAULLT and Sclavounos' theory for small amplitude
heave of an AR8 wing.
=#

using LiftingLineTheory
using PyPlot

let
    AR = 4
    wing = LiftingLineTheory.make_rectangular(StraightAnalyticWing, AR, AR)
    srf = 8
    k = srf / AR
    amp = 0.01
    omega = 2 * k
    dt = 0.015
    nsteps = 500

	
	println("Comparing LAULLT with Sclavounos in heave.")
	println("k = ", k)
	println("srf = ", srf)
	println("AR = ", AR)
	println("dt = ", dt)
	println("amp = ", amp)

    probs = HarmonicULLT(omega, wing)
    compute_collocation_points!(probs)
    compute_fourier_terms!(probs)
    cls = lift_coefficient(probs) * im * omega * amp
    figure()
    ts = collect(0:dt:dt*nsteps)
    clst = real.(cls .* exp.(im * omega * ts))
    plot(ts, clst, label="Sclavounos")
    ninner = 16
    bigsegs= range(-1, 1, length=ninner+1)
    innersolpos = collect((bigsegs[2:end]+bigsegs[1:end-1])./2)
    segs = collect(range(-1, 1, length=1*ninner+1))


    #prob = LAULLT(;kinematics=RigidKinematics2D(x->amp*cos(omega*x), x->0, 0.5),
    #    wing_planform=wing, dt=dt)
    #prob = LAULLT(;kinematics=RigidKinematics2D(x->0, x->deg2rad(5), 0),
    #    wing_planform=wing, dt=dt, segmentation=segmentation)
    prob = LAULLT(;kinematics=RigidKinematics2D(x->amp * cos(omega * x), x->0, 0.0),
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

    prob2d = LAUTAT(;kinematics=RigidKinematics2D(x->amp * cos(omega * x), x->0, 0.0), dt=dt)
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
    plot(rows[50:end, 1], rows[50:end, 7+4*7], label="LAULLT A0 8")
    plot(rows2d[50:end, 1], rows2d[50:end, 7], label="LAUTAT A0")
    plot(rows[50:end, 1], rows[50:end, 8], label="LAULLT A1 edge")
    plot(rows[50:end, 1], rows[50:end, 8+4*7], label="LAULLT A1 8")
    plot(rows2d[50:end, 1], rows2d[50:end, 8], label="LAUTAT A1")
    legend()
    return prob, rows, hdr
end

#=
A comparison of LAULLT and Sclavounos' theory for small amplitude
heave of an AR8 wing.
=#

using LiftingLineTheory
using PyPlot

let
    AR = 8
    wing = LiftingLineTheory.make_rectangular(StraightAnalyticWing, AR, AR)
    srf = 4
    k = srf / AR
    amp = 0.25#0.01
    omega = 2 * k
    dt = 0.015
    nsteps = 400
	
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

    prob = LAULLT(;kinematics=RigidKinematics2D(x->amp*cos(omega*x), x->0, 0),
        wing_planform=wing, dt=dt)
	println("n_inner = ", length(prob.inner_sols))
    
    hdr = csv_titles(prob)
    rows = zeros(0, length(hdr))
    print("\n")
    for i = 1 : nsteps
        print("\rStep ", i, " of ", nsteps, ".\t\t\t\t\t")
        advance_one_step(prob)
        rows = vcat(rows, csv_row(prob))
    end
    println("\n")
	println(size(rows))
    plot(rows[50:end, 1], rows[50:end, 5], label="LAULLT")

    xlabel("Time")
    ylabel("C_L")
    legend()
    return prob, rows, hdr
end

#=
Harmonic heave experiments with LAUTAT
=#

using LiftingLineTheory
using DelimitedFiles

let
    dt = 0.0125
    k = 1;
    amp = 1.;
    reg_rad = .001 * dt;
    oscl = 6;
    num_fourier_terms = 8;

    casename = "hOsc"*string(amp)*"_k"*string(k)*"_Reg"*string(reg_rad)*"_dt"*string(dt)*"_nt"*string(num_fourier_terms)
    casename = replace(casename, "." => "p")

	println("Case "*casename)
    println("dt = ", dt)
    println("amp = ", amp)
    println("k = ", k)
    println("reg_rad = ", reg_rad)
    println("num_fourier_terms = ", num_fourier_terms)

    omega = 2 * k
    period = 2 * pi / omega
    nsteps = Int64(ceil(period * oscl / dt))
    println("omega = ", omega)
    println("period = ", period)
    println("nsteps = ", nsteps)


    prob = LAUTAT(;kinematics=RigidKinematics2D(x->amp * cos(omega * x), x->0, 0.0),       # HEAVE
        dt=dt, num_fourier_terms=num_fourier_terms, reg_dist=reg_rad)
    
    hdr = csv_titles(prob)
    rows = zeros(0, length(hdr))
    for i = 1 : nsteps
        print("\rStep ", i, " of ", nsteps, ".\t\t\t\t\t")
        advance_one_step(prob)
        rows = vcat(rows, csv_row(prob))
    end
    to_vtk(prob, casename;
        updir=[0,0,1], translation=[-0.5,0,0])

    writefile = open(casename*".csv", "w")
    writedlm(writefile, hdr, ", ")
    writedlm(writefile, rows, ", ")
    close(writefile)
end

#=
Replicate Figure 6 from Ramesh 2013
=#

using LiftingLineTheory
using PyPlot

let
    rampfn = x->eldredge_ramp(x, 1, 3, 4, 6,  11, 1, 1)
    rampfnmax = rampfn(3.5)
    dtstar=0.015
    prob = LAUTAT(;
        kinematics=RigidKinematics2D(x->0, 
            x->deg2rad(25) * rampfn(x) / rampfnmax, -0.5),
        dt=dtstar,
        regularisation=singular_regularisation())

    hdr = csv_titles(prob)
    rows = zeros(0, length(hdr))

    nsteps = Int64(ceil(7/dtstar))
    for i = 1 : nsteps
        advance_one_step(prob)
        to_vtk(prob, "my_save_path_"*string(i))
        rows = vcat(rows, csv_row(prob))
    end

    figure()
    plot(rows[:, 1], rows[:, 9], "r-")
    ylabel("Cl")
    axis([0, 7, -2, 4])
    xlabel("Time")
    ax = gca()
    ax2 = twinx(ax)
    p = plot(rows[:, 1], rad2deg.(rows[:, 6]), "g-")
    axis([0, 7, -20, 40])
    ylabel("AoA")
end

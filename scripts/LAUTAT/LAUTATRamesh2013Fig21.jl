#=
Replicate Figure 6 from Ramesh 2013
=#

using LiftingLineTheory
using CVortex
using PyPlot

let
    rampfn = x->eldredge_ramp(x, 1, 3, 4, 6, 1, 1; a=11)
    rampfnmax = rampfn(3.5)
    dtstar=0.015
    prob = LAUTAT(;
        kinematics=RigidKinematics2D(x->0, 
            x->deg2rad(45) * rampfn(x) / rampfnmax, 0.5),
        dt=dtstar,
        regularisation=singular_regularisation())

    hdr = csv_titles(prob)
    rows = zeros(0, length(hdr))

    nsteps = Int64(ceil(7/dtstar))
    for i = 1 : nsteps
        advance_one_step(prob)
        #to_vtk(prob, "my_save_path_"*string(i))
        rows = vcat(rows, csv_row(prob))
    end

    figure()
    ax = gca()
    ax.imshow(imread("scripts/LAUTAT/Ramesh13ref/Fig21.PNG"), extent=[0, 7, -20, 60])
    ax.set_aspect("auto")
    p = plot(rows[:, 1], rad2deg.(rows[:, 6]), "g-", label="AoA")
    ylabel("AoA")
    axis([0, 7, -20, 60])
    ax2 = twinx(ax)
    plot(rows[:, 1], rows[:, 9], "r-", label="C_L")
    ylabel("Cl")
    axis([0, 7, -2, 6])
    xlabel("Time")
end

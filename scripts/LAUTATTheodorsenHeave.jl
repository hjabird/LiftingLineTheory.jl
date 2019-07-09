#=
Compare LAUTAT to theodorsen in heave.
=#

using LiftingLineTheory
using CVortex
using PyPlot

let
    ks = [0.1, 0.5, 1., 2.]
    dtstar=0.015
    amp = 0.01
    clengths = 40
    figure()
    for i in 1 : length(ks)
        omega = 2 * ks[i]
        hvfn = t->amp * cos(omega * t)
        prob = LAUTAT(;
            kinematics=RigidKinematics2D(hvfn, x->0, 0.0),
            dt=dtstar,
            regularisation=singular_regularisation())
        hdr = csv_titles(prob)
        rows = zeros(0, length(hdr))
        nsteps = Int64(ceil(20/dtstar))
        for i = 1 : nsteps
            advance_one_step(prob)
            rows = vcat(rows, csv_row(prob))
        end

        theod_amp = -2*pi*(theodorsen_fn(ks[i]) + im * ks[i] / 2) *
            im * omega * amp
        plot(rows[50:end, 1], rows[50:end, 9] ./ abs(theod_amp))
    end
    xlabel("Time")
    ylabel("C_L normalised with theodorsen CL")
end


push!(LOAD_PATH, "./src/")
using LiftingLineTheory
using Plots

let
    w = LiftingLineTheory.make_elliptic(StraightAnalyticWing, 4, 4)
    prob = LiftingLineTheory.TimeDomainULLT(x->deg2rad(5), w; pitch_plunge=5)
    LiftingLineTheory.compute_transfer_function!(prob)
    dt = 0.01

    ts = collect(0:0.1:9)
    cls = map(t->LiftingLineTheory.lift_coefficient_step(prob, t), ts)
    prob
end

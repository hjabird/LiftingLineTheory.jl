include("sclavounos.jl")

semispan = 2
aspect_ratio = 4
wing = make_rectangular(StraightAnalyticWing, aspect_ratio, semispan * 2)
srf = 8
area = wing_area(wing)

println("Semispan = ", semispan, ", AR = ", aspect_ratio, ", area = ", area)
println("SRF = ", srf, ", omega = ", srf/semispan)

y = collect(-semispan *0.99: semispan / 50 : semispan *0.99)
lcl_u = Vector{Complex{Float64}}(undef, length(y))
lcl_ps = Vector{Complex{Float64}}(undef, length(y))
lcl_eps = Vector{Complex{Float64}}(undef, length(y))


prob = HarmonicULLT(
    srf / semispan,
    wing,
    downwash_model = unsteady,
    pitch_plunge = 3
)
compute_collocation_points!(prob)
compute_fourier_terms!(prob)
lcl_u = map(x->chord_lift_coefficient(prob, x), y)
dCl_u = map(x->x[1] * wing.chord_fn(x[2]), zip(lcl_u, y))./ wing_area(wing)

prob.downwash_model = psuedosteady
compute_collocation_points!(prob)
compute_fourier_terms!(prob)
lcl_ps = map(x->chord_lift_coefficient(prob, x), y)
dCl_ps = map(x->x[1] * wing.chord_fn(x[2]), zip(lcl_ps, y))./ wing_area(wing)

prob.downwash_model = extpsuedosteady
compute_collocation_points!(prob)
compute_fourier_terms!(prob)
lcl_eps = map(x->chord_lift_coefficient(prob, x), y)
dCl_eps = map(x->x[1] * wing.chord_fn(x[2]), zip(lcl_eps, y))./ wing_area(wing)

prob.downwash_model = strip_theory
compute_collocation_points!(prob)
compute_fourier_terms!(prob)
lcl_st = map(x->chord_lift_coefficient(prob, x), y)
dCl_st = map(x->x[1] * wing.chord_fn(x[2]), zip(lcl_st, y)) ./ wing_area(wing)

using DataFrames
using CSVFiles


df = DataFrame(
    y_positions=y,
    strip=abs.(dCl_st),
    psuedosteady=abs.(dCl_ps),
    xfil=abs.(dCl_eps),
    unsteady=abs.(dCl_u))
save("reAR4_dCLy_k"*string(Int64(100*srf/4))*"_plunge_study.csv", df)



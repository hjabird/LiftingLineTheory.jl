include("sclavounos.jl")
using PyPlot

semispan = 2
aspect_ratio = 4
wing = make_elliptic(StraightAnalyticWing, aspect_ratio, semispan * 2)
srf = 1
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
    pitch_plunge = 5
)
compute_collocation_points!(prob)
compute_fourier_terms!(prob)
lcl_u = map(x->chord_lift_coefficient(prob, x), y)

prob.downwash_model = psuedosteady
compute_collocation_points!(prob)
compute_fourier_terms!(prob)
lcl_ps = map(x->chord_lift_coefficient(prob, x), y)

prob.downwash_model = extpsuedosteady
compute_collocation_points!(prob)
compute_fourier_terms!(prob)
lcl_eps = map(x->chord_lift_coefficient(prob, x), y)

figure()
function phase(x)
    return atan(imag(x)/ real(x)) * 180 / pi - 180
end

plot(y, abs.(lcl_u), label="unsteady")
plot(y, abs.(lcl_ps), label="psuedosteady")
plot(y, abs.(lcl_eps), label="extpsuedosteady")
title(L"Local lift abs (Rectangular AR4) $\omega d / U = 1$")
xlabel(L"y")
ylabel(L"ph(\frac{d}{dy} C_L)")
legend()

println("CL is ", compute_lift_coefficient(prob))

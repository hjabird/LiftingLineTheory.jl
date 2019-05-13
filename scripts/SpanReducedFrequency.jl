#
# SpanReducedFrequency.jl
#
# Copyright HJA Bird 2019
#
#	Studies responses of a wing oscillating at varying span reduced freq.
#
#============================================================================#

push!(LOAD_PATH, "../src")
using LiftingLineTheory

semispan = 2
aspect_ratio = 2
wing = LiftingLineTheory.make_van_dyke_cusped(StraightAnalyticWing, aspect_ratio, semispan*2, 0)
srf = collect(0.001:0.1:8.02)
println("Semispan = ", semispan, ", AR = ", aspect_ratio, ", area = ", area(wing))
kvar = 3
fterms = 8
kinem = y->1#y^2 / semispan^2

clst = Vector{Complex{Float64}}(undef, length(srf))
println("Starting")
for i = 1 : length(srf)
    srfv = srf[i]
    fq = srfv / semispan
    prob = HarmonicULLT(
        fq, wing;
        downwash_model = strip_theory,
        pitch_plunge = kvar,
        num_terms = fterms,
        amplitude_fn = kinem)
    compute_collocation_points!(prob)
    compute_fourier_terms!(prob)
    clst[i] = lift_coefficient(prob)
end
println("DONE strip")

cls = Vector{Complex{Float64}}(undef, length(srf))
for i = 1 : length(srf)
    srfv = srf[i]
    fq = srfv / semispan
    prob = HarmonicULLT(
        fq, wing;
        downwash_model = psuedosteady,
        pitch_plunge = kvar,
        num_terms = fterms,
        amplitude_fn = kinem)
    compute_collocation_points!(prob)
    compute_fourier_terms!(prob)
    cls[i] = lift_coefficient(prob)
end
println("DONE psuedosteady")

clt1 = Vector{Complex{Float64}}(undef, length(srf))
for i = 1 : length(srf)
    srfv = srf[i]
    fq = srfv / semispan
    prob = HarmonicULLT(
        fq, wing;
        downwash_model = streamwise_filaments,
        pitch_plunge = kvar,
        num_terms = fterms,
        amplitude_fn = kinem)
    compute_collocation_points!(prob)
    compute_fourier_terms!(prob)
    clt1[i] = lift_coefficient(prob)
end
println("DONE x fil")

clu = Vector{Complex{Float64}}(undef, length(srf))
for i = 1 : length(srf)
    srfv = srf[i]
    fq = srfv / semispan
    prob = HarmonicULLT(
        fq, wing;
        downwash_model = unsteady,
        pitch_plunge = kvar,
        num_terms = fterms,
        amplitude_fn = kinem)
    compute_collocation_points!(prob)
    compute_fourier_terms!(prob)
    clu[i] = lift_coefficient(prob)
end
println("DONE unsteady")

function oabs(x, srf)
    #return map(x->x[1]*abs(x[2]), zip(srf, x))
    return map(x->abs(x[2]), zip(srf, x))
end

using DataFrames
using CSVFiles
df = DataFrame(
    SpanReducedFrequency=srf,
    strip=abs.(clst),
    psuedosteady=abs.(cls),
    xfil=abs.(clt1),
    unsteady=abs.(clu))
save("elAR4_srf_plunge_study.csv", df)

#=
figure()
plot(srf, oabs(clst, srf), label="Strip theory")
plot(srf, oabs(cls, srf), label="Psuedosteady")
plot(srf, oabs(clt1, srf), label="Streamwise Filaments")
plot(srf, oabs(clu, srf), label="Unsteady")
axis([0, maximum(srf), 0, 8])
xlabel(L"\omega d/U")
ylabel(L"abs(C_L)")
legend()

figure()
plot(srf, map(x->atan(imag(x),real(x)) * 180/pi, clst), label="Strip theory")
plot(srf, map(x->atan(imag(x),real(x)) * 180/pi, cls), label="Psuedosteady")
plot(srf, map(x->atan(imag(x),real(x)) * 180/pi, clt1), label="Streamwise Filaments")
plot(srf, map(x->atan(imag(x),real(x)) * 180/pi, clu), label="Unsteady")
axis([0, maximum(srf), -270, 0])
xlabel(L"\omega d/U")
ylabel(L"ph(C_L)")
legend()
=#

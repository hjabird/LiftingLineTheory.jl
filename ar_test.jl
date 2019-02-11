include("sclavounos.jl")
using PyPlot
println("Starting AR test")

semispan = 4
aspect_ratio = collect(2 : 0.5 : 12)
shape = 0 # 0-rect, 1-elliptic, 2-lenticular, 3-cusped
srf = 1
fq = srf / semispan
kvar = 3
fterms = 8
kinem = y->1#y^2 / semispan^2

clst = Vector{Complex{Float64}}(undef, length(aspect_ratio))
println("Starting")
for i = 1 : length(aspect_ratio)
    ar = aspect_ratio[i]
    wing = make_van_dyke_cusped(StraightAnalyticWing, ar, semispan*2, shape)
    prob = HarmonicULLT(
        fq, wing;
        downwash_model = strip_theory,
        pitch_plunge = kvar,
        num_terms = fterms,
        amplitude_fn = kinem)
    compute_collocation_points!(prob)
    k_term3(prob, 0.1)
    compute_fourier_terms!(prob)
    clst[i] = compute_lift_coefficient(prob)
end
println("DONE strip")

cls = Vector{Complex{Float64}}(undef, length(aspect_ratio))

for i = 1 : length(aspect_ratio)
    ar = aspect_ratio[i]
    wing = make_van_dyke_cusped(StraightAnalyticWing, ar, semispan*2, shape)
    prob = HarmonicULLT(
        fq, wing;
        downwash_model = psuedosteady,
        pitch_plunge = kvar,
        num_terms = fterms,
        amplitude_fn = kinem)
    compute_collocation_points!(prob)
    k_term3(prob, 0.1)
    compute_fourier_terms!(prob)
    cls[i] = compute_lift_coefficient(prob)
end
println("DONE psuedosteady")

clt1 = Vector{Complex{Float64}}(undef, length(aspect_ratio))
for i = 1 : length(aspect_ratio)
    ar = aspect_ratio[i]
    wing = make_van_dyke_cusped(StraightAnalyticWing, ar, semispan*2, shape)
    prob = HarmonicULLT(
        fq, wing;
        downwash_model = extpsuedosteady,
        pitch_plunge = kvar,
        num_terms = fterms,
        amplitude_fn = kinem)
    compute_collocation_points!(prob)
    k_term3(prob, 0.1)
    compute_fourier_terms!(prob)
    clt1[i] = compute_lift_coefficient(prob)
end
println("DONE x fil")

clu = Vector{Complex{Float64}}(undef, length(aspect_ratio))
for i = 1 : length(aspect_ratio)
    ar = aspect_ratio[i]
    wing = make_van_dyke_cusped(StraightAnalyticWing, ar, semispan*2, shape)
    prob = HarmonicULLT(
        fq, wing;
        downwash_model = unsteady,
        pitch_plunge = kvar,
        num_terms = fterms,
        amplitude_fn = kinem)
    compute_collocation_points!(prob)
    k_term3(prob, 0.1)
    compute_fourier_terms!(prob)
    clu[i] = compute_lift_coefficient(prob)
end
println("DONE unsteady")

function oabs(x, srf)
    #return map(x->x[1]*abs(x[2]), zip(srf, x))
    return map(x->abs(x[2]), zip(srf, x))
end

figure()
plot(aspect_ratio, oabs(clst, aspect_ratio), label="Strip theory")
plot(aspect_ratio, oabs(cls, aspect_ratio), label="Psuedosteady")
plot(aspect_ratio, oabs(clt1, aspect_ratio), label="Streamwise Filaments")
plot(aspect_ratio, oabs(clu, aspect_ratio), label="Unsteady")
axis([0, maximum(aspect_ratio), 0, 8])
xlabel("Aspect ratio")
ylabel(L"abs(C_L)")
legend()

figure()
plot(aspect_ratio, map(x->atan(imag(x),real(x)) * 180/pi, clst), label="Strip theory")
plot(aspect_ratio, map(x->atan(imag(x),real(x)) * 180/pi, cls), label="Psuedosteady")
plot(aspect_ratio, map(x->atan(imag(x),real(x)) * 180/pi, clt1), label="Streamwise Filaments")
plot(srf, map(x->atan(imag(x),real(x)) * 180/pi, clu), label="Unsteady")
axisaspect_ratio0, maximum(srf), -270, 0])
xlabel("Aspect ratio")
ylabel(L"ph(C_L)")
legend()

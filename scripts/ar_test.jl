push!(LOAD_PATH, "../src")

using LiftingLineTheory

println("Starting AR test")

chord = 1
aspect_ratio = 1 ./ collect(0.5 : -0.05: 0.05)
shape = 1 # 0-rect, 1-elliptic, 2-lenticular, 3-cusped
fq = 2      * 2 / chord # chord reduced frequency
kvar = 3
fterms = 8
kinem = y->1#y^2 / semispan^2

clst = Vector{Complex{Float64}}(undef, length(aspect_ratio))
println("Starting")
for i = 1 : length(aspect_ratio)
    ar = aspect_ratio[i]
    wing = LiftingLineTheory.make_van_dyke_cusped(StraightAnalyticWing, ar, chord * ar, shape)
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

cls = Vector{Complex{Float64}}(undef, length(aspect_ratio))
for i = 1 : length(aspect_ratio)
    ar = aspect_ratio[i]
    wing = LiftingLineTheory.make_van_dyke_cusped(StraightAnalyticWing, ar, chord * ar, shape)
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

clt1 = Vector{Complex{Float64}}(undef, length(aspect_ratio))
for i = 1 : length(aspect_ratio)
    ar = aspect_ratio[i]
    wing = LiftingLineTheory.make_van_dyke_cusped(StraightAnalyticWing, ar, chord * ar, shape)
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

clu = Vector{Complex{Float64}}(undef, length(aspect_ratio))
for i = 1 : length(aspect_ratio)
    ar = aspect_ratio[i]
    wing = LiftingLineTheory.make_van_dyke_cusped(StraightAnalyticWing, ar, chord * ar, shape)
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

using DataFrames
using CSVFiles
df = DataFrame(
    aspectratio=aspect_ratio,
    strip=abs.(clst),
    psuedosteady=abs.(cls),
    xfil=abs.(clt1),
    unsteady=abs.(clu))

str = "elar_k"*string(Int64(fq * 50))*"_plunge_study.csv"
save(str, df)
println("Saved "*str)



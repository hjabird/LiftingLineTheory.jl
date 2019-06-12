
mutable struct SteadyLLT
    free_stream_vel :: Real
    aoa_fn :: Function
    wing :: StraightAnalyticWing
    lift_slope_fn :: Function

    num_terms ::Int64
    collocation_points :: Vector{Float64}
    fourier_terms :: Vector{Float64}
end

function compute_fourier_terms!(
    a :: SteadyLLT)

    k1 = -1 / pi 
    k2 = 1 / (4 * pi)
    c1 = (k, theta)->k1 * sin((2*k + 1) * theta) / a.wing.chord_fn(theta_to_y(theta))
    c2 = (k, theta)->k2 * (2*k + 1) * glauert_integral(k, theta)

    mat = map(
        (i, k)->c1(k, collocation_points[i]) + c2(k, collocation_points[i]),
        [(i, k) for i in 1 : num_terms, k in 1 : num_terms])
    aoa = map(x->aoa_fn, [i for i in 1:num_terms])
    solution = mat\aoa

    a.fourier_terms = solution
end


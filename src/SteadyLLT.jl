#
# SteadyULLT.jl
#
# Classical Prandtl ULLT right now. Might add curved/swept wing impl at some
# point?
#
# Copyright HJA Bird 2019
#
#==============================================================================#

mutable struct SteadyLLT
    wing :: StraightAnalyticWing
    U :: Real
    aoa_distribution :: Function
    fourier_terms :: Vector{Real}

    function SteadyLLT(wing :: StraightAnalyticWing; 
        free_stream_vel=1 :: Real,
        fourier_terms=16 :: Int,
        aoa_distribution=1 :: Union{Real, Function} )

        @assert(free_stream_vel > 0, 
            "Free stream velocity must be positive real.")
        @assert(isfinite(free_stream_vel), "Free stream vel must be finite.")

        if typeof(aoa_distribution)==Function
            @assert(hasmethod(aoa_dist, (Float64,)),
                "aoa_distribution function must be callable as "*
                "aoa_distribution(y) where y is a subtype of Real "*
                "(probably F64) and y is in [-semispan, semispan].")
            aoa_dist = aoa_distribution
        else
            aoa_dist = y->aoa_distribution
        end
        @assert(isfinite(aoa_dist(0.0)), "aoa distribution is not finite!")
        @assert(fourier_terms > 0, "Expecting a positive number of fourier"*
            " terms")
        return new(wing, free_stream_vel, aoa_dist, 
            zeros(fourier_terms))
    end
end

# Collocation points in theta space.
function compute_collocation_points(
    a :: SteadyLLT)

    nt = length(a.fourier_terms)
    @assert(nt > 0, "Number of terms in SteadyLLT must be more than 0")
    pos = Vector{Float64}(undef, nt)
    hpi = pi / 2
    for i = 1 : nt
        pos[i] = (pi * i - hpi) / (2 * nt)
    end
    return pos
end

# global spanwise coord to angular coord
function theta_to_y(
    a :: SteadyLLT,
    theta :: Real)
    # Yes
    @assert(0 <= theta <= pi)
    return a.wing.semispan * cos(theta)
end

function dtheta_dy(
    a :: SteadyLLT,
    y :: Real)
    # Yes
    @assert(abs(y) <= a.wing.semispan)
    result = -1 / sqrt(a.wing.semispan^2 - y^2)
    return result
end

function dsintheta_dy(
    a :: SteadyLLT,
    y :: Real,
    k :: Integer)
    # Yes
    @assert(k >= 0, "Positive wavenumbers only please!")
    theta = y_to_theta(a, y)
    dtdy = dtheta_dy(y)
    dGammadt = dsintheta_dtheta(a, theta, k)
    return dGammadt * dtdy
end

function dsintheta_dtheta(
    a :: SteadyLLT,
    theta :: Real,
    k :: Integer)
    # Yes
    @assert(k >= 0, "Positive wavenumbers only please!")
    @assert(0 <= theta <= pi)
    dGamma_dt = (2 * k + 1) * cos((2 * k + 1) * theta)
    return dGamma_dt
end

function y_to_theta(
    a :: SteadyLLT,
    y :: Real)
    # Yes
    @assert(abs(y) <= a.wing.semispan)
    return acos(y / a.wing.semispan)
end

function compute_fourier_terms!(a::SteadyLLT)
    col_points = compute_collocation_points(a)
    alphas = map(theta->a.aoa_distribution(theta_to_y(a, theta)), col_points)
    chords = map(theta->a.wing.chord_fn(theta_to_y(a, theta)), col_points)
    nt = length(a.fourier_terms)
    @assert(nt > 0)
    
    direct_mat = zeros(nt, nt)
    for i = 1:nt
        for j = 1:nt
            direct_mat[i, j] = - sin((2*j-1) * col_points[i]) / 
                                                (pi * chords[i] * a.U)
        end
    end

    integro_mat = zeros(nt, nt)
    for i = 1:nt
        for j = 1:nt
            integro_mat[i, j] = -(2 * j - 1) * sin((2 * j - 1) * col_points[i]) /
                (4 * a.wing.semispan * a.U * sin(col_points[i]))
        end
    end

    gammas = (integro_mat + direct_mat) \ (-alphas)
    a.fourier_terms = gammas
    return
end

function lift_coefficient(a::SteadyLLT, y::Real)
    @assert(abs(y) <= a.wing.semispan, "abs(y) should be smaller than the wing semispan")
    bv = bound_vorticity(a, y)
    c = a.wing.chord_fn(y)
    U = a.U
    lift_coeff = bv / (0.5 * U * c)
    return lift_coeff
end

function lift_coefficient(a::SteadyLLT)
    s = a.wing.semispan
    points, weights = FastGaussQuadrature.gausslegendre(40)
    points, weights = linear_remap(points, weights, -1, 1, -s, s)
    function integrand(y)
        return lift_coefficient(a, y) * a.wing.chord_fn(y)
    end
    integral = sum(weights .* integrand.(points))
    coeff = integral / area(a.wing)
    return coeff
end

function drag_coefficient(a::SteadyLLT, y::Real)
    @assert(false, "Not yet implemented")
end
function drag_coefficient(a::SteadyLLT)
    @assert(false, "Not yet implemented")
end
function moment_coefficient(a::SteadyLLT, y::Real)
    @assert(false, "Not yet implemented")
end
function moment_coefficient(a::SteadyLLT)
    @assert(false, "Not yet implemented")
end

function bound_vorticity(a::SteadyLLT, y::Real)
    @assert(abs(y) <= a.wing.semispan)
    theta = y_to_theta(a, y)
    s = a.fourier_terms[1] * sin(theta)
    for i = 2 : length(a.fourier_terms)
        s += a.fourier_terms[i] * sin( (2 * i - 1) * theta )
    end
    return s
end

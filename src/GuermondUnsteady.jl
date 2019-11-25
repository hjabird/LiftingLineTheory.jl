#
# GuermondUnsteady.jl
#
# Rectangular wings only for now!
#
# Copyright HJA Bird 2019
#
#==============================================================================#

mutable struct GuermondUnsteady
    angular_fq :: Real              # in rad / s
    free_stream_vel :: Real

    wing :: StraightAnalyticWing
    amplitude_fn :: Function        # Amplitude of oscillation wrt/ span pos.
    pitch_plunge :: Int64           # Plunge = 3, Pitch = 5. Otherwise invalid.

    chord_len_scale :: Float64
end

function Y_to_y(a::GuermondUnsteady, Y::Real)
    return Y / a.wing.semispan
end

function Y_to_y(a::GuermondUnsteady, Y::Vector{<:Real})
    return Y ./ a.wing.semispan
end

function y_to_Y(a::GuermondUnsteady, y::Real)
    return y * a.wing.semispan
end

function y_to_Y(a::GuermondUnsteady, y::Vector{<:Real})
    return y .* a.wing.semispan
end

function compute_chord_len_scale!(a::GuermondUnsteady)
    a = area(a.wing)
    s = a.wing.semispan
    c = a/(2*s)
    a.chord_len_scale = c
    return
end

function X_to_x(a::GuermondUnsteady, X::Real)
    return X / a.chord_len_scale
end

function X_to_x(a::GuermondUnsteady, X::Vector{<:Real})
    return X ./ a.chord_len_scale
end

function upwash_distribution(a::GuermondUnsteady, Y::Real; order::Int=1)
    @assert(abs(Y) < a.wing.semispan)
    @assert(order >= 0, "Positive order required")
    @assert(order < 2, "Only programmed to order 1!")
    bv = NaN
    U = a.free_stream_vel
    chord = a.wing.chord_fn(Y)
    k = a.angular_fq * chord / (2* a.free_stream_vel)
    amp = a.amplitude_fn(Y)
    # 0th order: 2D
    if a.pitch_plunge == 3
        uw = make_plunge_function(HarmonicUpwash2D, amp, k;
            free_stream_vel = a.free_stream_vel, semichord=chord/2)
    elseif a.pitch_plunge == 5
        uw = make_pitch_function(HarmonicUpwash2D, amp, k;
            free_stream_vel = a.free_stream_vel, semichord=chord/2)
    end
    # 1st order: 3D 
    if order > 0
        ar = aspect_ratio(a.wing)
        pd_coeff = operator_K1_excl_exp_term(a, Y)
        uws = make_sinusoidal_gust_function(HarmonicUpwash2D, # THIS BIT I SHOULD BE CAREFUL ABOUT...
            -pd_coeff, k, U, semichord=chord / a.chord_len_scale)
        uw = uw - uws / ar
    end
    return uw
end

# Currently assumes rectangular wing!
function operator_K1_excl_exp_term(a::GuermondUnsteady, Y::Real)
    y = Y_to_y(a, Y)
    nu = a.angular_fq * a.wing.semispan / a.free_stream_vel
    k =  a.angular_fq * a.chord_len_scale / a.free_stream_vel
    c = a.wing.chord_fn(y)
    G = bound_vorticity(a, y; order=0) * exp(im * k * c / 2)

    # The nasty to compute constant bit
    t1 = 1 / (2*pi*(y^2 - 1))
    # See https://www.cs.uaf.edu/~bueler/M611heaviside.pdf
    t2 = -im * nu / (2*pi*(y^2 - 1)) * 1 / (im * nu)
    
    # t3 requires numerical integration
    function integrand(v::Real)
        t1 = exp(im * nu * v) / v
        t21 = sqrt(v^2 + (y-1)^2) / (y-1)
        t22 = -sqrt(v^2 + (y+1)^2) / (y+1)
        t23 = 2 
        t24 = 2 * v / (y^2 - 1)
        return t1 * (t21 + t22 + t23 + t24)
    end
    t3 = -im * nu / (4 * pi) * QuadGK.quadgk(integrand, -Inf, 0)[1]
    constpart = t1 + t2 + t3

    # The sinusoidal inner solution downwash's coefficient
    coeff = G * constpart
    return coeff
end

function bound_vorticity(a::GuermondUnsteady, Y::Real; order::Int=1)
    @assert(abs(Y) < a.wing.semispan)
    @assert(order >= 0, "Positive order required")
    @assert(order < 2, "Only programmed to order 1!")
    bv = NaN
    uw = upwash_distribution(a, Y; order=order)
    bv = bound_vorticity(uw)
    return bv
end

function lift_coefficient(a::GuermondUnsteady, Y::Real; order=1)
    uw = upwash_distribution(a, Y; order=order)
    cL = lift_coefficient(uw)
    return cL
end

function lift_coefficient(a::GuermondUnsteady; order=1)
    s = a.wing.semispan
    points, weights = FastGaussQuadrature.gausslegendre(40)
    points, weights = linear_remap(points, weights, -1, 1, -s, s)
    CL = sum(weights .* map(Y->lift_coefficient(a, Y; order=order), points))
    return CL
end


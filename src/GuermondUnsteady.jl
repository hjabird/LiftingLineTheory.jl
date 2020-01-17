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
    span_len_scale :: Float64
    guermond_k :: Float64
    guermond_nu :: Float64

    function GuermondUnsteady(
        angular_fq::Real, wing::StraightAnalyticWing;
        free_stream_vel ::Real = 1, amplitude_fn::Function=x->1,
        pitch_plunge::Int=3, chord_len_scale::Real=NaN,
        span_len_scale::Real=NaN, guermond_k::Real=NaN,
        guermond_nu::Real=NaN)

        @assert(wing.chord_fn(0) == wing.chord_fn(1e-6), "Rect. only")
        chord_len_scale = (chord_len_scale == chord_len_scale ?
            chord_len_scale : wing.chord_fn(0))
        span_len_scale = (span_len_scale == span_len_scale ?
            span_len_scale : wing.semispan)
        guermond_k = (guermond_k == guermond_k ?
            guermond_k : angular_fq * chord_len_scale / free_stream_vel)
        guermond_nu = (guermond_nu == guermond_nu ?
            guermond_nu : angular_fq * span_len_scale / free_stream_vel)

        return new(Float64(angular_fq), Float64(free_stream_vel), wing, 
            amplitude_fn, pitch_plunge, chord_len_scale, span_len_scale,
            guermond_k, guermond_nu)
    end
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

function upwash_distribution(a::GuermondUnsteady, Y::Real; order::Int=1)
    #= The upwash distribution is a combination of the 2D upwash and, if
    order = 1, the 3D correction. Input are given with respect to unscaled
    coordinates and the output is also with respect to unscaled coordinates =#

    @assert(abs(Y) < a.wing.semispan)
    @assert(order >= 0, "Positive order required")
    @assert(order < 2, "Only programmed to order 1!")
    bv = NaN
    U = a.free_stream_vel
    chord = a.chord_len_scale # Should equal chord for rect wing.
    k = a.angular_fq * chord / (2* a.free_stream_vel) # Not the same as guermond_k
    amp = a.amplitude_fn(Y)
    # 0th order: 2D
    if a.pitch_plunge == 3
        uw = make_plunge_function(HarmonicUpwash2D, amp, k;
            free_stream_vel = a.free_stream_vel, semichord=chord/2)
    elseif a.pitch_plunge == 5
        uw = make_pitch_function(HarmonicUpwash2D, amp, k;
            free_stream_vel = a.free_stream_vel, semichord=chord/2)
    end
    # 1st order: 3D - the more complicated bit. 
    if order > 0 # or == 1
        ar = aspect_ratio(a.wing)
        pd_coeff = operator_K1_excl_exp_term(a, Y)
        uws = make_sinusoidal_gust_function(HarmonicUpwash2D,
            pd_coeff, k; free_stream_vel= U, semichord=chord/2)
        #uws = make_plunge_function(HarmonicUpwash2D, pd_coeff / a.angular_fq, k;
        #    free_stream_vel = a.free_stream_vel, semichord=chord/2)
        uw = uw - uws/ ar
    end
    return uw
end

# Currently assumes rectangular wing!
function operator_K1_excl_exp_term(a::GuermondUnsteady, Y::Real)
    y = Y_to_y(a, Y)
    @assert(abs(y) < 1)
    nu = a.guermond_nu
    k =  a.guermond_k
    C = a.chord_len_scale
    # The bound vorticity in G&S is calculated according to the normalised
    # problem, but here we are using Kussner-Schwarz to do the work so 
    # we must divide through by U. I think.
    G = bound_vorticity(a, y; order=0) * exp(im * k * C / 2) / a.free_stream_vel

    # The nasty to compute constant bit
    t1 = 1 / (2 * pi)
    t21 = 1 / (y^2 - 1)
    
    t221 = - im * nu / 2
    # t22 requires numerical integration
    function integrand(v::Real)
        #tk1 = 1 / v
        #tk21 = sqrt(v^2 + (y-1)^2) / (y-1)
        #tk22 = -sqrt(v^2 + (y+1)^2) / (y+1)
        #tk23 = 2
        #tk24 = 2 * v / (y^2 - 1)
        #tk0 = tk1 * (tk21 + tk22 + tk23 + tk24)
        #Float64(tk0)
        #return tk0
        # *exp(im * nu * v) oscl part in quadrature
        ti21 = sqrt(1/v^2 + 1/(y-1)^2)
        ti22 = sqrt(1/v^2 + 1/(y+1)^2)
        ti23 = 2 / v
        ti3 = 2 / (y^2 - 1)
        return  ti21 + ti22 + ti23 + ti3
    end
    t222 = fejer_quadrature_exp(integrand, nu, -Inf, -1e-8; finite=true)
    @assert(isfinite(t222), "Term 3 (the integral) isn't finite!")
    constpart = t1 * (t21 + t221 * t222)
    #println("At k = ", k, ": t1 = ", t1*t21,", t2 = ", abs(t1 * t221 * t222))

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
    points, weights = FastGaussQuadrature.gausslegendre(100)
    points, weights = linear_remap(points, weights, -1, 1, -s, s)
    lctc = Y->lift_coefficient(a, Y; order=order) * a.wing.chord_fn(Y)
    CL = sum(weights .* map(lctc, points)) / area(a.wing)
    return CL
end


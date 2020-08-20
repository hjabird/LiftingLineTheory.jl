#
# HarmonicULLT2.jl
#
# A harmonic ULLT where the principal of downwash decomposition from Guermond
# and Sellier's 1991 paper has been used, applied to uniform downwash. All 
# of the integrals  are based upon Bessel/Struve functions.
#
# Doesn't work right now!
#
# Interface is identical to that in HarmonicULLT.jl
#
# Copyright HJA Bird 2019-2020
#
#==============================================================================#

mutable struct HarmonicULLT2
    angular_fq :: Real              # in rad / s
    free_stream_vel :: Real

    wing :: StraightAnalyticWing
    amplitude_fn :: Function        # Amplitude of oscillation wrt/ span pos.
    pitch_plunge :: Int64           # Plunge = 3, Pitch = 5. Otherwise invalid.

    downwash_model :: DownwashModel # See DownwashModel defintion
    num_terms :: Int64              # Number of terms in fourier expansion
    fourier_terms :: Vector{Complex{Real}}
    collocation_points :: Vector{Real}  # In terms of theta in [0, pi]

    function HarmonicULLT2(
        angular_fq :: Real,
        wing :: StraightAnalyticWing;
        free_stream_vel = 1,
        amplitude_fn = y -> 1,
        pitch_plunge = 3,
        downwash_model = unsteady,
        num_terms = 8,
        fourier_terms = Vector{Float64}(undef, 1),
        collocation_points = Vector{Float64}(undef, 1)
    )
       @assert(angular_fq >= 0, "Positive frequencies only please") 
       @assert(isfinite(angular_fq), "Non-finite frequencies cannot be used")
       @assert(wing.semispan > 0, "Wing must have a positive span")
       @assert(wing.chord_fn(0) >= 0, "Wing must have positive chord")

       new(angular_fq, free_stream_vel, wing, amplitude_fn, pitch_plunge,
        downwash_model, num_terms, fourier_terms, collocation_points)
    end
end

function d_heave(
    a :: HarmonicULLT2,
    y :: Real)

    @assert(abs(y) <= a.wing.semispan)
    norm_fq = a.angular_fq / a.free_stream_vel
    semichord = a.wing.chord_fn(y) / 2
    num = 4 * a.free_stream_vel * exp(-im * semichord * norm_fq) * 
        a.amplitude_fn(y)
    den = im * SpecialFunctions.hankelh2(0, norm_fq * semichord) +
        SpecialFunctions.hankelh2(1, norm_fq * semichord)
    return num / den
end

function d_heave_normalised(
    a :: HarmonicULLT2,
    y :: Real)
    # Amplitude fn = 1
    @assert(abs(y) <= a.wing.semispan)
    norm_fq = a.angular_fq / a.free_stream_vel
    semichord = a.wing.chord_fn(y) / 2
    num = 4 * a.free_stream_vel * exp(-im * semichord * norm_fq)
    den = im * SpecialFunctions.hankelh2(0, norm_fq * semichord) +
        SpecialFunctions.hankelh2(1, norm_fq * semichord)
    return num / den
end

function d_pitch(
    a :: HarmonicULLT2,
    y :: Real)

    @assert(abs(y) <= a.wing.semispan)
    U = a.free_stream_vel
    om = a.angular_fq
    nu = om / U
    semichord = a.wing.chord_fn(y) / 2
    num = -4 * U * exp(-im * semichord * nu) * 
        a.amplitude_fn(y) * (U/(im * om) + semichord / 2)
    den = im * SpecialFunctions.hankelh2(0, nu * semichord) +
        SpecialFunctions.hankelh2(1, nu * semichord)
    return num / den
end

function compute_collocation_points!(
    a :: HarmonicULLT2)

    nt = a.num_terms
    @assert(nt > 0, "Number of terms in HarmonicULLT2 must be more than 0")
    pos = Vector{Float64}(undef, a.num_terms)
    #hpi = pi / 2
    #for i = 1 : nt
    #    pos[i] = (pi * i - hpi) / (2 * nt)
    #end
    for i = 1 : nt
        pos[i] = i * (pi / 2) / (nt + 1)
    end
    a.collocation_points = pos
    return
end

function theta_to_y(
    a :: HarmonicULLT2,
    theta :: Real)
    # Yes
    @assert(0 <= theta <= pi)
    return a.wing.semispan * cos(theta)
end

function dtheta_dy(
    a :: HarmonicULLT2,
    y :: Real)
    # Yes
    @assert(abs(y) <= a.wing.semispan)
    result = -1 / sqrt(a.wing.semispan^2 - y^2)
    return result
end

function dsintheta_dy(
    a :: HarmonicULLT2,
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
    a :: HarmonicULLT2,
    theta :: Real,
    k :: Integer)
    # Yes
    @assert(k >= 0, "Positive wavenumbers only please!")
    @assert(0 <= theta <= pi)
    dGamma_dt = (2 * k + 1) * cos((2 * k + 1) * theta)
    return dGamma_dt
end

function y_to_theta(
    a :: HarmonicULLT2,
    y :: Real)
    # Yes
    @assert(abs(y) <= a.wing.semispan)
    return acos(y / a.wing.semispan)
end

function integrate_gammaprime_k_streamwise(
    a :: HarmonicULLT2,
    y :: Real,
    k :: Integer)

    @assert(k >= 0)
    @assert( abs(y) <= a.wing.semispan )
    
    if( a.downwash_model == unsteady )
        integral = integrate_gammaprime_k_streamwise_fil(a, y, k)
    elseif( a.downwash_model == pseudosteady )
        integral = integrate_gammaprime_k_pseudosteady(a, y, k)
    elseif( a.downwash_model == streamwise_filaments )
        integral = integrate_gammaprime_k_streamwise_fil(a, y, k)
    elseif( a.downwash_model == strip_theory )
        integral = 0
    end
    return integral
end

function integrate_gammaprime_k_spanwise(
    a :: HarmonicULLT2,
    y :: Real,
    k :: Integer)

    @assert(k >= 0)
    @assert( abs(y) <= a.wing.semispan )
    
    if( a.downwash_model == unsteady )
        integral = integrate_gammaprime_k_spanwise_fil(a, y, k)
    elseif( a.downwash_model == pseudosteady )
        integral = 0
    elseif( a.downwash_model == streamwise_filaments )
        integral = 0
    elseif( a.downwash_model == strip_theory )
        integral = 0
    end
    return integral
end

function integrate_gammaprime_k_streamwise_pseudosteady(
    a :: HarmonicULLT2, 
    y :: Real, 
    k :: Integer)

    theta = y_to_theta(a, y)
    integral = (2*k + 1) * pi * sin((2*k + 1) * theta) / 
        (2 * a.wing.semispan * sin(theta))
    return integral
end

function integrate_gammaprime_k_streamwise_fil(
    a :: HarmonicULLT2,
    y :: Real,
    k :: Integer)

    theta_singular = y_to_theta(a, y)
    v = a.angular_fq / a.free_stream_vel
    s = a.wing.semispan
    function non_singular(dy)
        sgn = dy > 0 ? 1 : -1
        co = v*dy
        t1 = v*abs(dy) != 0 ? v * abs(dy) * SpecialFunctions.besselk(1,v*abs(dy)) : 1
        t2 = 1/2 * v * abs(dy) * im * pi * (
            SpecialFunctions.besseli(1, v*abs(dy)) - struve_l(-1, v*dy))
        return t1 + t2
    end
    ssm_var = non_singular(0)
    function integrand(theta_0)
        dy = y - theta_to_y(a, theta_0)
        sgn = dy > 0 ? 1 : -1
        sing = s * cos((2*k+1)*theta_0) / dy
        # We remove the singularity here and need to add it back later.
        ns = non_singular(dy)
        return sing * (ns - ssm_var)
    end

    nodes1, weights1 = FastGaussQuadrature.gausslegendre(70)   
    pts2 = map(
        x->linear_remap(x[1], x[2], -1, 1, theta_singular, pi),
        zip(nodes1, weights1))
    pts1 = map(
        x->linear_remap(x[1], x[2], -1, 1, 0, theta_singular),
        zip(nodes1, weights1))
    integral =
        sum(last.(pts1) .* map(integrand, first.(pts1))) +
        sum(last.(pts2) .* map(integrand, first.(pts2))) 
    coeff = -(2*k + 1) / (4 * pi * s)
    ret = coeff * (integral 
        - ssm_var * pi * sin((2* k + 1) * theta_singular) / sin(theta_singular))
    return ret
end

#= ORINGINAL IMPL
function integrate_gammaprime_k_spanwise_fil(
    a :: HarmonicULLT2,
    y :: Real,
    k :: Integer)

    theta_sing = y_to_theta(a, y) 
    gamma_local = sin((2 * k + 1) * theta_sing)
    v = a.angular_fq / a.free_stream_vel
    s = a.wing.semispan

    # Off the wing tips:
    function tip_dw(dy)
        local t1 = -im * gamma_local * v / ( 2 * pi )
        local t21 = im * pi * SpecialFunctions.besseli(0, v * abs(dy)) / 2
        local t22 = SpecialFunctions.besselk(0, v * abs(dy))
        local t23 = -pi * im * struve_l(0, v * abs(dy)) / 2
        return t1 * (t21 + t22 + t23)
    end
    tips_effect = tip_dw(s - y) + tip_dw(s + y)

    # Over the span:
    function integrand(y0)
        gamma_diff = sin((2 * k + 1) * y_to_theta(a, y0)) - gamma_local
        dy = y - y0
        co = - im * v / (4*pi)
        va = v * abs(dy)
        t1 = 1 / abs(dy)
        t2 = v * pi /2 * (SpecialFunctions.besseli(0, va) - struve_l(0, va))
        t3 = im * v * SpecialFunctions.besselk(0, va)
        return gamma_diff * co * (t1 + t2 + t3)
    end

    nodes1, weights1 = FastGaussQuadrature.gausslegendre(70)   
    pts2 = map(
        x->linear_remap(x[1], x[2], -1, 1, y, s),
        zip(nodes1, weights1))
    pts1 = map(
        x->linear_remap(x[1], x[2], -1, 1, -s, y),
        zip(nodes1, weights1))
    integral =
        sum(last.(pts1) .* map(integrand, first.(pts1))) +
        sum(last.(pts2) .* map(integrand, first.(pts2))) 
    
    return - tips_effect - integral 
end
=#
using PyPlot
function integrate_gammaprime_k_spanwise_fil(
    a :: HarmonicULLT2,
    y :: Real,
    k :: Integer)

    theta_sing = y_to_theta(a, y) 
    gamma_local = sin((2 * k + 1) * theta_sing)
    v = a.angular_fq / a.free_stream_vel
    s = a.wing.semispan

    coeff = - im * v / (4 * pi)

    function integrand_inspan(eta)
        local delta_gamma = sin( (2 * k + 1) * y_to_theta(a, eta)) - gamma_local
        local vay = v * abs(y - eta)
        local ti11 = pi * v * (y - eta) / 2
        local ti12 = SpecialFunctions.besseli(0, vay) - struve_l(0, vay)
        local ti2 = im * v * (y-eta) * SpecialFunctions.besselk(0, vay)
        local ti3 = - (y > eta ? 1 : -1);
        return delta_gamma * (ti11 * ti12 + ti2 + ti3) / (y - eta)
    end
    nodes1, weights1 = FastGaussQuadrature.gausslegendre(50)   
    pts2 = map(
        x->linear_remap(x[1], x[2], -1, 1, y, s),
        zip(nodes1, weights1))
    pts1 = map(
        x->linear_remap(x[1], x[2], -1, 1, -s, y),
        zip(nodes1, weights1))
    integral_inspan =
        sum(last.(pts1) .* map(integrand_inspan, first.(pts1))) +
        sum(last.(pts2) .* map(integrand_inspan, first.(pts2))) 

    function nonsing_part_outer(eta)
        local delta_gamma = -gamma_local
        local vay = v * abs(y - eta)
        local ti11 = pi * v * (y - eta) / 2
        local ti12 = SpecialFunctions.besseli(0, vay) - struve_l(0, vay)
        local ti2 = im * v * (y-eta) * SpecialFunctions.besselk(0, vay)
        local ti3 = - (y > eta ? 1 : -1);
        return delta_gamma * (ti11 * ti12 + ti2 + ti3) * (y-eta)
    end
    non_singular_ps = nonsing_part_outer( s)
    non_singular_ms = nonsing_part_outer(-s)
    function integrand_outspan_ms(eta)
        return (nonsing_part_outer(eta) - non_singular_ms) / (y - eta)^2 # Since we're integrating to infinity
    end
    function integrand_outspan_ps(eta)
        return (nonsing_part_outer(eta) - non_singular_ps) / (y - eta)^2
    end
    nodes1, weights1 = FastGaussQuadrature.gausslaguerre(5)   
    nodes2 = map(x->x+s, nodes1)
    nodes1 = map(x->-(x+s), nodes1)
    #integral_outspan, ~ = QuadGK.quadgk(integrand_outspan, -Inf, -s; rtol=1e-3) +
     #   QuadGK.quadgk(integrand_outspan, s, Inf; rtol=1e-5) +
    #    1 / (s-y) + 1 / (s + y)
    #    sum(weights1 .* map(integrand_outspan, nodes1)) +
    #    sum(weights1 .* map(integrand_outspan, nodes2)) 

    ysm = collect(-4:0.1:-2)
    plot(ysm, map(y->real(coeff * integrand_outspan_ms(y)), ysm), label="k=0.1")
    plot(ysm, map(y->imag(coeff * integrand_outspan_ms(y)), ysm), label="k=0.1")
    ysp = collect(2:0.1:4);
    plot(ysp, map(y->real(coeff * integrand_outspan_ps(y)), ysp), label="k=0.1")
    plot(ysp, map(y->imag(coeff * integrand_outspan_ps(y)), ysp), label="k=0.1")

    return coeff * (integral_inspan + integral_outspan); 
end

function gamma_terms_matrix(
    a :: HarmonicULLT2 )
    idxs = collect(1:a.num_terms)
    mtrx = map(
        i->sin((2 * i[2] - 1) * a.collocation_points[i[1]]),
        collect((j,k) for j in idxs, k in idxs)
    )
    return mtrx
end

function rhs_vector(
    a :: HarmonicULLT2 )

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "Sclavounos.jl: HarmonicULLT2.pitch_plunge must be 3" *
        " (plunge) or 5 (pitch). Value was ", a.pitch_plunge, "." )
    if(a.pitch_plunge == 3) # Plunge
        d_fn = d_heave
    elseif(a.pitch_plunge == 5)   # Pitch
        d_fn = d_pitch
    end
    circ_vect = map(
        theta->d_fn(a, theta_to_y(a, theta)),
        a.collocation_points
    )
    return circ_vect
end

function integro_diff_mtrx_coeff(
    a :: HarmonicULLT2,
    y_pos :: Real)

    coeff = d_heave_normalised(a, y_pos) / (a.angular_fq * im)
    return coeff
end

function compute_fourier_terms!(
    a :: HarmonicULLT2 )

    @assert(length(a.collocation_points)>1, "Only one collocation point. "*
        "Did you call compute_collocation_points!(a::HarmonicULLT2)?")
    gamma_mtrx = gamma_terms_matrix(a)
    integro_diff_mtrx = 
        map(
            in->
                integro_diff_mtrx_coeff(a, theta_to_y(a, a.collocation_points[in[1]+1])) * 
                (integrate_gammaprime_k_streamwise(a,  theta_to_y(a, a.collocation_points[in[1]+1]), in[2])
                + integrate_gammaprime_k_spanwise(a,  theta_to_y(a, a.collocation_points[in[1]+1]), in[2])),
            collect((i, j) for i in 0:a.num_terms-1, j in 0:a.num_terms-1)
        )
    rhs_vec = rhs_vector(a)
    solution = (gamma_mtrx - integro_diff_mtrx) \ rhs_vec
    a.fourier_terms = solution
    return
end    

function lift_coefficient(
    a :: HarmonicULLT2 )

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "HarmonicULLT2.pitch_plunge must equal 3 (plunge) or 5 (pitch).")

    w_area = area(a.wing)
    integrand = y->lift_coefficient(a, y) * 
        a.amplitude_fn(y) * a.wing.chord_fn(y)
    nodes, weights = FastGaussQuadrature.gausslegendre(70)
    pts = map(
        x->linear_remap(x[1], x[2], -1, 1, -a.wing.semispan, a.wing.semispan),
        zip(nodes, weights))
    integral = sum(last.(pts) .* map(integrand, first.(pts)))/ w_area
    CL = integral
    return CL
end

function lift_coefficient(
    a :: HarmonicULLT2,
    y :: Real)

    # Notes 5 pg 57
    if(a.pitch_plunge == 3)
        associated_cl_fn = associated_chord_cl_heave
    elseif(a.pitch_plunge == 5)
        associated_cl_fn = associated_chord_cl_pitch
    end
    clA = a.amplitude_fn(y) * associated_cl_fn(a, y) - f_eq(a, y) *
        associated_chord_cl_heave(a, y)
    return clA / a.amplitude_fn(y)
end

function associated_chord_cl_heave(
    a :: HarmonicULLT2,
    y :: Real)

    # Notes 5 pg 53
    @assert(abs(y) <= a.wing.semispan)
    k = a.angular_fq * a.wing.chord_fn(y) / (2 * a.free_stream_vel)
    t1 = -2 * pi * theodorsen_fn(k) 
    t2 = -im * pi * k 
    return t1 + t2
end

function associated_chord_cl_pitch(
    a :: HarmonicULLT2,
    y :: Real)

    # Notes #7 pg 4
    @assert(abs(y) <= a.wing.semispan)
    semichord = a.wing.chord_fn(y) / 2 
    U = a.free_stream_vel
    omega = a.angular_fq
    k = omega * a.wing.chord_fn(y) / (2 * U)
    t1 = pi * semichord
    t21 = theodorsen_fn(k) * pi
    t22 = semichord - 2 * U * im  / omega
    t2 = t21 * t22
    return t1 + t2
end

function moment_coefficient(
    a :: HarmonicULLT2 )

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "HarmonicULLT2.pitch_plunge must equal 3 (plunge) or 5 (pitch).")

    w_area = area(a.wing)
    semispan = a.wing.semispan
    integrand = y->moment_coefficient(a, y) * 
        a.amplitude_fn(y) * a.wing.chord_fn(y)^2
    nodes, weights = FastGaussQuadrature.gausslegendre(70)
    pts = map(
        x->linear_remap(x[1], x[2], -1, 1, -semispan, semispan),
        zip(nodes, weights))
    integral = sum(last.(pts) .* map(integrand, first.(pts))) / 
        (w_area^2 / (2*semispan))
    CM = integral
    return CM
end

function moment_coefficient(
    a :: HarmonicULLT2,
    y :: Real)

    # Notes 6 pg 55
    if(a.pitch_plunge == 3)
        associated_cm_fn = associated_chord_cm_heave
    elseif(a.pitch_plunge == 5)
        associated_cm_fn = associated_chord_cm_pitch
    end
    cmA = a.amplitude_fn(y) * associated_cm_fn(a, y) - f_eq(a, y) *
        associated_chord_cm_heave(a, y)
    return cmA / a.amplitude_fn(y)
end

function associated_chord_cm_heave(
    a :: HarmonicULLT2,
    y :: Real)

    # Notes #7 pg 4
    @assert(abs(y) <= a.wing.semispan)
    @assert(a.angular_fq > 0)
    k = a.angular_fq * a.wing.chord_fn(y) / (2 * a.free_stream_vel)
    num = - pi * theodorsen_fn(k)
    den = 2
    return num / den
end

function associated_chord_cm_pitch(
    a :: HarmonicULLT2,
    y :: Real)
    #Notes #7 pg. 4
    @assert(abs(y) <= a.wing.semispan)
    @assert(a.angular_fq > 0)
    chord = a.wing.chord_fn(y)
    semichord = chord/2
    omega = a.angular_fq
    U = a.free_stream_vel
    k = omega * a.wing.chord_fn(y) / (2 * U)
    Ck = theodorsen_fn(k)
    t1 = -pi / 4
    t21 = im * k * semichord/ 4
    t22 = Ck * (2 * im * U / omega - semichord)
    t23 = semichord
    t2 = t21 + t22 + t23
    t = t1 * t2
    return t
end

function drag_coefficient(
    a :: HarmonicULLT2)

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "HarmonicULLT2.pitch_plunge must equal 3 (plunge) or 5 (pitch).")

    w_area = area(a.wing)
    integrand = y->drag_coefficient(a, y) * 
        a.amplitude_fn(y) * a.wing.chord_fn(y)
    nodes, weights = FastGaussQuadrature.gausslegendre(70)
    pts = map(
        x->linear_remap(x[1], x[2], -1, 1, -a.wing.semispan, a.wing.semispan),
        zip(nodes, weights))
    integral = sum(last.(pts) .* map(integrand, first.(pts)))/ w_area
    CD = integral
    return CD
end

function drag_coefficient(
    a :: HarmonicULLT2,
    y :: Real)

    @assert(abs(y) <= a.wing.semispan)
    @assert(a.angular_fq > 0)
    # Based on Ramesh 2013
    cs = 2 * pi * a0_term(a, y)^2
    cd = -cs
    if a.pitch_plunge == 5
        cdp = lift_coefficient(a, y) * a.amplitude_fn(y)
        cd += cdp
    end
    return cd / a.amplitude_fn(y)
end

function a0_term(
    a :: HarmonicULLT2,
    y :: Real)

    @assert(abs(y) <= a.wing.semispan)
    @assert(a.angular_fq > 0)

    if(a.pitch_plunge == 3)
        associated_a0_fn = associated_a0_term_heave
    elseif(a.pitch_plunge == 5)
        associated_a0_fn = associated_a0_term_pitch
    end
    a0_term = a.amplitude_fn(y) * associated_a0_fn(a, y) - f_eq(a, y) *
        associated_a0_term_heave(a, y)
    return a0_term
end

function associated_a0_term_pitch(
    a :: HarmonicULLT2,
    y :: Real)

    chord = a.wing.chord_fn(y)
    semichord = chord/2
    omega = a.angular_fq
    U = a.free_stream_vel
    k = omega * a.wing.chord_fn(y) / (2 * U)
    Ck = theodorsen_fn(k)
    t1 = Ck * (1 - im * omega * chord / (4 * U))
    t2 = - im * omega * chord / (4 * U)
    return t1 + t2
end

function associated_a0_term_heave(
    a :: HarmonicULLT2,
    y :: Real)

    chord = a.wing.chord_fn(y)
    semichord = chord/2
    omega = a.angular_fq
    U = a.free_stream_vel
    k = omega * a.wing.chord_fn(y) / (2 * U)
    Ck = theodorsen_fn(k)
    t1 = - Ck * im * omega / U
    return t1
end

function f_eq(
    a :: HarmonicULLT2,
    y :: Real)
    # Notes 5 pg 57
    @assert(abs(y) <= a.wing.semispan)
    if(a.downwash_model == strip_theory)
        f = 0
    else
        if(a.pitch_plunge == 3)
            d_fn = d_heave
        elseif(a.pitch_plunge == 5)
            d_fn = d_pitch
        end
        f = (d_fn(a, y) - bound_vorticity(a, y)) / d_heave_normalised(a, y)
    end
    return f
end

function bound_vorticity(
    a :: HarmonicULLT2,
    y :: Real)

    @assert(abs(y) <= a.wing.semispan)
    @assert(a.num_terms >= 1)

    theta = y_to_theta(a, y)
    sum = sin(theta) * a.fourier_terms[1]
    for i = 2:a.num_terms
        @assert(isfinite(sum))
        sum += sin((2 * i - 1) * theta) * a.fourier_terms[i]
    end
    @assert(isfinite(sum))
    return sum
end

# END HarmonicULLT2.jl
#============================================================================#

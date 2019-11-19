#
# HarmonicULLT3.jl
#
# Copyright HJA Bird 2019
#
#==============================================================================#

using PyPlot

mutable struct HarmonicULLT3
    angular_fq :: Real              # in rad / s
    free_stream_vel :: Real

    wing :: StraightAnalyticWing
    amplitude_fn :: Function        # Amplitude of oscillation wrt/ span pos.
    pitch_plunge :: Int64           # Plunge = 3, Pitch = 5. Otherwise invalid.

    downwash_model :: DownwashModel # See DownwashModel defintion
    num_terms :: Int64              # Number of terms in fourier expansion
    fourier_terms :: Vector{Complex{Real}}
    collocation_points :: Vector{Real}  # In terms of theta in [0, pi]

    function HarmonicULLT3(
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
    a :: HarmonicULLT3,
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
    a :: HarmonicULLT3,
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
    a :: HarmonicULLT3,
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
    a :: HarmonicULLT3)

    nt = a.num_terms
    @assert(nt > 0, "Number of terms in HarmonicULLT3 must be more than 0")
    pos = Vector{Float64}(undef, a.num_terms)
    hpi = pi / 2
    for i = 1 : nt
        pos[i] = (pi * i - hpi) / (2 * nt)
    end
    a.collocation_points = pos
    return
end

function theta_to_y(
    a :: HarmonicULLT3,
    theta :: Real)
    # Yes
    @assert(0 <= theta <= pi)
    return a.wing.semispan * cos(theta)
end

function dtheta_dy(
    a :: HarmonicULLT3,
    y :: Real)
    # Yes
    @assert(abs(y) <= a.wing.semispan)
    result = -1 / sqrt(a.wing.semispan^2 - y^2)
    return result
end

function dsintheta_dy(
    a :: HarmonicULLT3,
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
    a :: HarmonicULLT3,
    theta :: Real,
    k :: Integer)
    # Yes
    @assert(k >= 0, "Positive wavenumbers only please!")
    @assert(0 <= theta <= pi)
    dGamma_dt = (2 * k + 1) * cos((2 * k + 1) * theta)
    return dGamma_dt
end

function y_to_theta(
    a :: HarmonicULLT3,
    y :: Real)
    # Yes
    @assert(abs(y) <= a.wing.semispan)
    return acos(y / a.wing.semispan)
end

function integrate_gammaprime_k_streamwise(
    a :: HarmonicULLT3,
    y :: Real,
    k :: Integer)

    @assert(k >= 0)
    @assert( abs(y) <= a.wing.semispan )
    
    if( a.downwash_model == unsteady )
        integral = integrate_gammaprime_k_streamwise_fil(a, y, k)
    elseif( a.downwash_model == psuedosteady )
        integral = integrate_gammaprime_k_psuedosteady(a, y, k)
    elseif( a.downwash_model == streamwise_filaments )
        integral = integrate_gammaprime_k_streamwise_fil(a, y, k)
    elseif( a.downwash_model == strip_theory )
        integral = 0
    end
    return integral
end

function integrate_gammaprime_k_spanwise(
    a :: HarmonicULLT3,
    y :: Real,
    k :: Integer)

    @assert(k >= 0)
    @assert( abs(y) <= a.wing.semispan )
    
    if( a.downwash_model == unsteady )
        integral = integrate_gammaprime_k_spanwise_fil(a, y, k)
    elseif( a.downwash_model == psuedosteady )
        integral = 0
    elseif( a.downwash_model == streamwise_filaments )
        integral = 0
    elseif( a.downwash_model == strip_theory )
        integral = 0
    end
    return integral
end

function integrate_gammaprime_k_streamwise_psuedosteady(
    a :: HarmonicULLT3, 
    y :: Real, 
    k :: Integer)

    theta = y_to_theta(a, y)
    integral = (2*k + 1) * pi * sin((2*k + 1) * theta) / 
        (2 * a.wing.semispan * sin(theta))
    return integral
end

using PyPlot
function integrate_gammaprime_k_streamwise_fil(
    a :: HarmonicULLT3,
    y :: Real,
    k :: Integer)

    theta_singular = y_to_theta(a, y)
    v = a.angular_fq / a.free_stream_vel
    s = a.wing.semispan
    function integrand(theta_0)
        dy = y - theta_to_y(a, theta_0)
        sgn = dy > 0 ? 1 : -1
        t1 = - dsintheta_dtheta(a, theta_0, k) / (8 * pi)
        # We remove the singularity here and need to add it back later.
        t2 = 2*(v*sgn*SpecialFunctions.besselk(1,v*abs(dy)) - 1/dy)
        t3 = sgn * im * pi * v * (struve_l(-1, v*dy) - 
            SpecialFunctions.besseli(0, v*abs(dy)))                 # Term 3 temporarily removed!_---------------
        return -10.5 * t1 * (t2 ) / (2*k + 1)
    end

    thetas = collect(0:0.01:pi)
    its = integrand.(thetas)
    plot(thetas, its)

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
    coeff = (2*k+1) / (2 * s)
    return coeff * (integral + 1/(4*s) * sin((2* k + 1) * theta_singular) / sin(theta_singular))
end

function integrate_gammaprime_k_spanwise_fil(
    a :: HarmonicULLT3,
    y :: Real,
    k :: Integer)

    println("Integrating spanwise!")
    theta_sing = y_to_theta(a, y) 
    gamma_local = sin((2 * k + 1) * theta_sing)
    v = a.angular_fq / a.free_stream_vel
    s = a.wing.semispan
    coeff = -im * v * gamma_local / (4 * pi)

    # Off the wing tips:
    tipf = dy->im * pi * (
            SpecialFunctions.besseli(0, v*dy) - struve_l(0, v * dy)) / 2 + 
            SpecialFunctions.besselk(0, v*dy)
    tips_effect = coeff * (tipf(s - y) + tipf(s + y))

    # Over the span:
    function integrand(y0)
        gamma_diff = sin((2 * k + 1) * y_to_theta(a, y0)) - gamma_local
        dy = y - y0
        k0 = v * dy^2
        t1 = im * v * k0 * (struve_l(0,k0) - SpecialFunctions.besseli(0,k0)) / 
            8 * abs(dy)
        t2 = im * v / (4 * pi * abs(dy))
        t3 = - k0 * v * SpecialFunctions.besselk(0,k0) / (4 * pi * dy)
        return t1 + t2 + t3
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
    
    return integral + tips_effect
end

function gamma_terms_matrix(
    a :: HarmonicULLT3 )
    idxs = collect(1:a.num_terms)
    mtrx = map(
        i->sin((2 * i[2] - 1) * a.collocation_points[i[1]]),
        collect((j,k) for j in idxs, k in idxs)
    )
    return mtrx
end

function rhs_vector(
    a :: HarmonicULLT3 )

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "Sclavounos.jl: HarmonicULLT3.pitch_plunge must be 3" *
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
    a :: HarmonicULLT3,
    y_pos :: Real)

    coeff = d_heave_normalised(a, y_pos) / (2 * pi * a.angular_fq * im)
    return coeff
end

function compute_fourier_terms!(
    a :: HarmonicULLT3 )

    @assert(length(a.collocation_points)>1, "Only one collocation point. "*
        "Did you call compute_collocation_points!(a::HarmonicULLT3)?")
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
    a :: HarmonicULLT3 )

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "HarmonicULLT3.pitch_plunge must equal 3 (plunge) or 5 (pitch).")

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
    a :: HarmonicULLT3,
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
    a :: HarmonicULLT3,
    y :: Real)

    # Notes 5 pg 53
    @assert(abs(y) <= a.wing.semispan)
    k = a.angular_fq * a.wing.chord_fn(y) / (2 * a.free_stream_vel)
    t1 = -2 * pi * theodorsen_fn(k) 
    t2 = -im * pi * k 
    return t1 + t2
end

function associated_chord_cl_pitch(
    a :: HarmonicULLT3,
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
    a :: HarmonicULLT3 )

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "HarmonicULLT3.pitch_plunge must equal 3 (plunge) or 5 (pitch).")

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
    a :: HarmonicULLT3,
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
    a :: HarmonicULLT3,
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
    a :: HarmonicULLT3,
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
    a :: HarmonicULLT3)

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "HarmonicULLT3.pitch_plunge must equal 3 (plunge) or 5 (pitch).")

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
    a :: HarmonicULLT3,
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
    a :: HarmonicULLT3,
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
    a :: HarmonicULLT3,
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
    a :: HarmonicULLT3,
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
    a :: HarmonicULLT3,
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
    a :: HarmonicULLT3,
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

# END HarmonicULLT3.jl
#============================================================================#

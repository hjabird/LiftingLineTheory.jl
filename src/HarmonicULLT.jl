#
# HarmonicULLT.jl
#
# A Prandtl like solution to unsteady lifting-line problems that assumes 
# uniform downwash over the wing.
#
# Several methods can be used according to choice of downwash model.
# downwash_model=
#   strip_theory: the 3D correction is ignored.
#   psuedosteady: the 3D correction assumes psuedosteady, like Prandtl.
#   streamwise_filaments: the 3D correction only include the streamwise 
#       vorticity (it oscillates!), but not the wake's spansiwse vorticity.
#   unsteady: the 3D correction includes all elements of the wake's vorticity
#       using Sclavounos' kernel (but not identical method).
#
# Use:
#   wing = make_elliptic(StraightAnalyticWing, 4, 4)
#   prob = HarmonicULLT(1, wing; downwash_model=psuedosteady)
#   compute_collocation_points!(prob)
#   compute_fourier_terms!(prob)
#   lift_coefficient(prob)
#   moment_coefficient(prob)
#   bound_vorticity(prob, 0.1) # at y is 0.1
#   lift_coefficient(prob, 0.1) # at y is 0.1
#
# Copyright HJA Bird 2019-2020
#
#==============================================================================#

mutable struct HarmonicULLT
    angular_fq :: Real              # in rad / s
    free_stream_vel :: Real

    wing :: StraightAnalyticWing
    amplitude_fn :: Function        # Amplitude of oscillation wrt/ span pos.
    pitch_plunge :: Int64           # Plunge = 3, Pitch = 5. Otherwise invalid.

    downwash_model :: DownwashModel # See DownwashModel defintion
    num_terms :: Int64              # Number of terms in fourier expansion
    fourier_terms :: Vector{Complex{Real}}
    collocation_points :: Vector{Real}  # In terms of theta in [0, pi]

    function HarmonicULLT(
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

# 2D bound vorticity due to heave (include amplitude, chord, etc.)
function d_heave(
    a :: HarmonicULLT,
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

# 2D bound vorticity for a unit heave motion (include chord)
function d_heave_normalised(
    a :: HarmonicULLT,
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

# 2D bound vorticity due to pitch (include amplitude, chord, etc.)
function d_pitch(
    a :: HarmonicULLT,
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

# Compute collocation points according to a method that produces good results.
function compute_collocation_points!(
    a :: HarmonicULLT)

    nt = a.num_terms
    @assert(nt > 0, "Number of terms in HarmonicULLT must be more than 0")
    pos = Vector{Float64}(undef, a.num_terms)
    hpi = pi / 2
    for i = 1 : nt
        pos[i] = (pi * i - hpi) / (2 * nt)
    end
    a.collocation_points = pos
    return
end

# Coordinate transforms and derivatives ========================================
function theta_to_y(
    a :: HarmonicULLT,
    theta :: Real)
    # Yes
    @assert(0 <= theta <= pi)
    return a.wing.semispan * cos(theta)
end

function dtheta_dy(
    a :: HarmonicULLT,
    y :: Real)
    # Yes
    @assert(abs(y) <= a.wing.semispan)
    result = -1 / sqrt(a.wing.semispan^2 - y^2)
    return result
end

function dsintheta_dy(
    a :: HarmonicULLT,
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
    a :: HarmonicULLT,
    theta :: Real,
    k :: Integer)
    # Yes
    @assert(k >= 0, "Positive wavenumbers only please!")
    @assert(0 <= theta <= pi)
    dGamma_dt = (2 * k + 1) * cos((2 * k + 1) * theta)
    return dGamma_dt
end

function y_to_theta(
    a :: HarmonicULLT,
    y :: Real)
    # Yes
    @assert(abs(y) <= a.wing.semispan)
    return acos(y / a.wing.semispan)
end

# Compute the integrals for Sclavounos' K ======================================
function k_term1_singularity(
    a :: HarmonicULLT,
    delta_y :: Real) 
    # Yes
    @assert(delta_y != 0, "delta_y == 0 leads to NaN (inf) answer")
    return 1 / delta_y
end

function k_term1_numerator(
    a :: HarmonicULLT,
    delta_y :: Real)
    # Yes
    return exp(- (a.angular_fq / a.free_stream_vel) * abs(delta_y)) / 2
end

function k_term2(
    a :: HarmonicULLT,
    delta_y :: Real)
    # Yes
    coeff = sign(delta_y) / 2
    nu = a.angular_fq / a.free_stream_vel
    e1_term = - im * nu * expint(nu * abs(delta_y))
    return coeff * e1_term
end

function k_term3(
    a :: HarmonicULLT,
    delta_y :: Real)
    # Yes
    coeff = sign(delta_y) / 2
    nu = a.angular_fq / a.free_stream_vel
    p = nu * p_eq(a, nu * abs(delta_y))

    return coeff * p
end

function p_eq(
    a :: HarmonicULLT,
    delta_y :: Real)
    # Is correct. See notes #7 pg.6 or #1 pg.63 or #1 pg. 76.
    # Eq. 3.21 in Sclavounos1987. Checked vs. mathematica.
    function integrand1(t :: T) where T <: Real
        val = - delta_y * exp(-delta_y * t) * (asin(1 / t) + sqrt(t^2 - 1) - t)
        return val / exp(-(t-1))    # Because of the quadrature
    end
    function integrand2(t :: T) where T <: Real
        return exp(-delta_y * t) * (sqrt(1 - t^2) - 1) / t
    end

    points1, weights1 = FastGaussQuadrature.gausslaguerre(30) # laGUERRE
    points2, weights2 = FastGaussQuadrature.gausslegendre(20) # leGENDRE
    pts2 = map(
        x->linear_remap(x[1], x[2], -1, 1, 0, 1),
        zip(points2, weights2))
    points1 .+= 1
    term1 = -exp(-delta_y) * (pi/2 - 1) - sum(weights1 .* map(integrand1, points1))
    term2 = sum(last.(pts2) .* map(integrand2, first.(pts2)))
    return term1 + im * term2
end

#= DIRECT EVLN of integral with Guass-Laguerre
function p_eq(
    a :: HarmonicULLT,
    delta_y :: Real)
    # Is correct. See notes #7 pg.6 or #1 pg.63 or #1 pg. 76.
    # Eq. 3.21 in Sclavounos1987.
    function integrand1(t :: T) where T <: Real
        val = exp(-delta_y * t) * (sqrt(t^2 - 1) - t)/t
        return val / exp(-(t-1))    # Because of the quadrature
    end
    function integrand2(t :: T) where T <: Real
        return exp(-delta_y * t) * (sqrt(1 - t^2) - 1) / t
    end

    points1, weights1 = FastGaussQuadrature.gausslaguerre(40) # laGUERRE
    points2, weights2 = FastGaussQuadrature.gausslegendre(20) # leGENDRE
    pts2 = map(
        x->linear_remap(x[1], x[2], -1, 1, 0, 1),
        zip(points2, weights2))
    term1 = sum(weights1 .* map(integrand1, points1 .+ 1))
    term2 = sum(last.(pts2) .* map(integrand2, first.(pts2)))
    return term1 + im * term2
end
=#

function integrate_gammaprime_k_term1(
    a :: HarmonicULLT,
    y :: Real,
    k :: Integer)
    # Pretty sure this is right.
    # Yes
    theta_singular = y_to_theta(a, y)
    
    # We're using the singularity subtraction method to deal with a CPV problem.
    singularity_coefficient = 
        dsintheta_dtheta(a, theta_singular, k) * k_term1_numerator(a, 0)
    
    function integrand(theta_0 :: Real)
        eta = theta_to_y(a, theta_0)
        singular_part = k_term1_singularity(a, y - eta)
        nonsingular_K = k_term1_numerator(a, y - eta)
        gamma_dtheta = dsintheta_dtheta(a, theta_0, k)
        
        singular_subtraction = (nonsingular_K  * gamma_dtheta - 
                                                    singularity_coefficient)
        return singular_part * singular_subtraction
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
        sum(last.(pts2) .* map(integrand, first.(pts2))) +
        singularity_coefficient * 0. # Glauert integral
    return -integral
end

#= ORIGINAL: using expint singularity =#
function integrate_gammaprime_k_term2(
    a :: HarmonicULLT,
    y :: Real,
    k :: Integer)
    # Confirmed against Mathematica evaluation of integrals.
    
    @assert(abs(y) < a.wing.semispan)
    @assert(k >= 0)
    theta_singular = y_to_theta(a, y)
    v = a.angular_fq / a.free_stream_vel
    s = a.wing.semispan
    
    integral_coefficient = im * v / 2
    function nonsingular_integrand(theta_0)
        return (2*k+1) * cos((2*k+1)*theta_0) / (v * s * sin(theta_0))
    end
    function singular_integrand(theta_0)
        return v * s * sin(theta_0) *
            expint(v * s * abs(cos(theta_singular) - cos(theta_0)))
    end
    
    # The singular part (in terms of the singularity subtraction method) of the integral
    singular_integral = v * (y + s) * expint(v * (y + s)) +
                        v * (y - s) * expint(v * (s - y)) -
                        exp(-v * (y + s)) +
                        exp(-v * (s - y))
        
    ssm_variable = nonsingular_integrand(theta_singular)
    function numerical_integrand(theta_0)
        singular_var = singular_integrand(theta_0)
        non_singular_var = nonsingular_integrand(theta_0)
        singularity_subtraction = non_singular_var - ssm_variable
        integrand = singular_var * singularity_subtraction
        return integrand
    end
    
    nodes1, weights1 = FastGaussQuadrature.gausslegendre(70)
    pts2 = map(
        x->linear_remap(x[1], x[2], -1, 1, theta_singular, pi),
        zip(nodes1, weights1))
    pts1 = map(
        x->linear_remap(x[1], x[2], -1, 1, 0, theta_singular),
        zip(nodes1, weights1))
    
    int_lower = sum(last.(pts1) .* map(numerical_integrand, first.(pts1)))
    int_upper = sum(last.(pts2) .* map(numerical_integrand, first.(pts2)))
    complete_integral = integral_coefficient * (int_upper - int_lower + ssm_variable * singular_integral)
    return complete_integral
end

#= Extracting the logarithmic singularity from expint
function integrate_gammaprime_k_term2(
    a :: HarmonicULLT,
    y :: Real,
    k :: Integer)
    
    @assert(abs(y) < a.wing.semispan)
    @assert(k >= 0)
    theta_singular = y_to_theta(a, y)
    v = a.angular_fq / a.free_stream_vel
    s = a.wing.semispan
    
    integral_multiplier = im * v * (2*k + 1) / 2
    function integral_1(ts)
        eta = s * cos(ts)
        return sign(y - eta) * cos((2 * k + 1) * ts) *
            (expint(v * abs(y-eta)) + log(v * abs(y-eta)))
    end
    function integral_2_sing(ts)
        eta = s * cos(ts)
        return v * s * sin(ts) * log(v * abs(y - eta))
    end
    function integral_2_nsing(ts)
        eta = s * cos(ts)
        return sign(y - eta) * cos((2*k + 1) * ts) / (v * s * sin(ts))
    end
    singularity_mult = integral_2_nsing(acos(y/s))
    function integral_2_sing_sub(ts)
        return integral_2_sing(ts) * (integral_2_nsing(ts) - singularity_mult)
    end
    singularity_integral =  v * (y + s) * log( v * (y + s)) -
                            v * (y - s) * log(-v * (y - s)) -
                            v * (y + s) +
                            v * (y - s)

    nodes1, weights1 = FastGaussQuadrature.gausslegendre(70)
    pts2 = map(
        x->linear_remap(x[1], x[2], -1, 1, theta_singular, pi),
        zip(nodes1, weights1))
    pts1 = map(
        x->linear_remap(x[1], x[2], -1, 1, 0, theta_singular),
        zip(nodes1, weights1))
    
    int1_lower = sum(last.(pts1) .* map(integral_1, first.(pts1)))
    int1_upper = sum(last.(pts2) .* map(integral_1, first.(pts2)))
    int2_lower = sum(last.(pts1) .* map(integral_2_sing_sub, first.(pts1)))
    int2_upper = sum(last.(pts2) .* map(integral_2_sing_sub, first.(pts2)))
    complete_integral = integral_multiplier * (int1_upper + int1_lower - 
                    int2_upper - int2_lower - singularity_mult * singularity_integral)
    return complete_integral
end=#

function integrate_gammaprime_k_term3(
    a :: HarmonicULLT,
    y :: Real,
    k :: Integer)
    
    @assert(k >= 0)
    @assert(abs(y) < a.wing.semispan)

    # Don't touch - correct.
    theta_singular = y_to_theta(a, y)
    function integrand(theta_0)
        eta = theta_to_y(a, theta_0)
        return dsintheta_dtheta(a, theta_0, k) * k_term3(a, y - eta)
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
    return -integral
end

# 3D integral for psuedo-steady problem ========================================
function integrate_gammaprime_k_psuedosteady(
    a :: HarmonicULLT, 
    y :: Real, 
    k :: Integer)

    theta = y_to_theta(a, y)
    integral = (2*k + 1) * pi * sin((2*k + 1) * theta) / 
        (2 * a.wing.semispan * sin(theta))
    return integral
end

# 3D integral for oscillating streamwise filaments =============================
function integrate_gammaprime_k_streamwise_fil(
    a :: HarmonicULLT,
    y :: Real,
    k :: Integer)

    theta_singular = y_to_theta(a, y)
    ssm_var = integrate_gammaprime_k_streamwise_fil_subint(a, 0, k)
    function integrand(theta_0)
        eta = theta_to_y(a, theta_0)
        singular = a.wing.semispan * cos((2*k +1)*theta_0) / (y - eta) # DOES THIS NEED at 1 / 2 in it?
        non_singular = 
            integrate_gammaprime_k_streamwise_fil_subint(a, y - eta, k)
        return singular * (non_singular - ssm_var)
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
    coeff = - (2*k + 1) / (2 * a.wing.semispan)
    return coeff * (integral -
        ssm_var * pi * sin((2* k + 1) * theta_singular) / sin(theta_singular))
end

function integrate_gammaprime_k_streamwise_fil_subint(
    a :: HarmonicULLT,
    y :: Real,
    k :: Integer)

    om = a.angular_fq
    function integrand(t :: Real)
        num = exp(im * om * y * t / a.free_stream_vel)
        den = (t^2 + 1)^(3/2)
        return num / den * exp(t)    # Because of the quadrature
    end
    # Can fiddle with the Laguerre quadrature so that the numerator here fits
    # better?
    points, weights = FastGaussQuadrature.gausslaguerre(50)
    integral = sum(weights .* integrand.(points))
    return integral
end

# The shared part of the solution ==============================================
# 3D integral method selection.
function integrate_gammaprime_k(
    a :: HarmonicULLT,
    y :: Real,
    k :: Integer)

    @assert(k >= 0)
    @assert( abs(y) <= a.wing.semispan )
    
    if( a.downwash_model == unsteady )
        i1 = integrate_gammaprime_k_term1(a, y, k)  # Don't touch - correct.
        i2 = integrate_gammaprime_k_term2(a, y, k)  # Don't touch - correct.
        i3 = integrate_gammaprime_k_term3(a, y, k)  # Don't touch - correct.
        integral = i1 + i2 + i3
    elseif( a.downwash_model == psuedosteady )
        integral = integrate_gammaprime_k_psuedosteady(a, y, k)
    elseif( a.downwash_model == streamwise_filaments )
        integral = integrate_gammaprime_k_streamwise_fil(a, y, k)
    elseif( a.downwash_model == strip_theory )
        integral = 0
    end
    return integral
end

# the Gamma = sin(j * acos(y_i)) matrix
function gamma_terms_matrix(
    a :: HarmonicULLT )
    idxs = collect(1:a.num_terms)
    mtrx = map(
        i->sin((2 * i[2] - 1) * a.collocation_points[i[1]]),
        collect((j,k) for j in idxs, k in idxs)
    )
    return mtrx
end

# 2D contributions
function rhs_vector(
    a :: HarmonicULLT )

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "Sclavounos.jl: HarmonicULLT.pitch_plunge must be 3" *
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
    a :: HarmonicULLT,
    y_pos :: Real)

    coeff = d_heave_normalised(a, y_pos) / (2 * pi * a.angular_fq * im)
    return coeff
end

# Solve!
function compute_fourier_terms!(
    a :: HarmonicULLT )

    @assert(length(a.collocation_points)>1, "Only one collocation point. "*
        "Did you call compute_collocation_points!(a::HarmonicULLT)?")
    gamma_mtrx = gamma_terms_matrix(a)
    integro_diff_mtrx = 
        map(
            in->
                integro_diff_mtrx_coeff(a, theta_to_y(a, a.collocation_points[in[1]+1])) * 
                integrate_gammaprime_k(a,  theta_to_y(a, a.collocation_points[in[1]+1]), in[2]),
            collect((i, j) for i in 0:a.num_terms-1, j in 0:a.num_terms-1)
        )
    rhs_vec = rhs_vector(a)
    solution = (gamma_mtrx - integro_diff_mtrx) \ rhs_vec
    a.fourier_terms = solution
    return
end    

# Post-processing ==============================================================
function lift_coefficient(
    a :: HarmonicULLT )

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "HarmonicULLT.pitch_plunge must equal 3 (plunge) or 5 (pitch).")

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
    a :: HarmonicULLT,
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
    a :: HarmonicULLT,
    y :: Real)

    # Notes 5 pg 53
    @assert(abs(y) <= a.wing.semispan)
    k = a.angular_fq * a.wing.chord_fn(y) / (2 * a.free_stream_vel)
    t1 = -2 * pi * theodorsen_fn(k) 
    t2 = -im * pi * k 
    return t1 + t2
end

function associated_chord_cl_pitch(
    a :: HarmonicULLT,
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
    a :: HarmonicULLT )

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "HarmonicULLT.pitch_plunge must equal 3 (plunge) or 5 (pitch).")

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
    a :: HarmonicULLT,
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
    a :: HarmonicULLT,
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
    a :: HarmonicULLT,
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
    a :: HarmonicULLT)

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "HarmonicULLT.pitch_plunge must equal 3 (plunge) or 5 (pitch).")

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
    a :: HarmonicULLT,
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

# We can in theory extract the a0 term from the theodorsen and see if we violate
# Ramesh et al.'s LESP criterion (although we can't shed LEVs).
function a0_term(
    a :: HarmonicULLT,
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
    a :: HarmonicULLT,
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
    a :: HarmonicULLT,
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

# The matching function F that corrects for 3D effects.
function f_eq(
    a :: HarmonicULLT,
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
    a :: HarmonicULLT,
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

# END HarmonicULLT.jl
#============================================================================#

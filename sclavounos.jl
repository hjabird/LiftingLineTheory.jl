
import SpecialFunctions

@enum DownwashModel begin
    psuedosteady = 1
    extpsuedosteady = 2
    unsteady = 3
end

mutable struct SclavounosULLT
    angular_fq :: Real
    free_stream_vel :: Real
    semispan :: Real

    chord_at_points :: Function
    amplitude_fn :: Function
    pitch_plunge :: Int64           # Plunge = 3, Pitch = 5. Otherwise invalid.

    downwash_model :: Function
    num_terms :: Int64
    terms :: Vector{Complex{Real}}
    collocation_points :: Vector{Real}  # In terms of theta in [0, pi]
end

function d3(
    a :: SclavounosULLT,
    chord :: Real)

    norm_fq = a.angular_fq / a.free_stream_vel
    semichord = chord / 2
    num = 4 * U * exp(-im * semichord * norm_fq)
    den = im * SpecialFunctions.hankelh2(0, norm_fq * semichord) +
        SpecialFunctions.hankelh2(1, norm_fq * semichord)
    return num / den
end

function d5(
    a :: SclavounosULLT,
    chord :: Real)

    return -chord * sclavounos_d3(SclavounosULLT, chord) / 4
end

function compute_collocation_points(
    a :: SclavounosULLT)

    nt = a.num_terms
    assert(nt > 0, "Number of terms in SclavounosULLT must be more than 0")
    pos = Vector{Float64}(undef, a.num_terms)
    hpi = pi / 2
    for i = 1 : nt
        pos[i] = (pi * i - hpi) / (2 * nt)
    end
    return
end

function theta_to_y(
    a :: SclavounosULLT,
    theta :: Real)

    assert(abs(theta) < pi / 2)
    return a.semispan * cos(theta)
end

function dtheta_dy(
    a :: SclavounosULLT,
    y :: Real)
    
    assert(abs(y) <= a.semispan)
    result = -1 / sqrt(pow(a.semispan, 2) - y^2)
    return result
end

function dsintheta_dy(
    a :: SclavounosULLT,
    y :: Real,
    k :: Int)
    
    theta = y_to_theta(a, y)
    dtdy = dtheta_dy(y)
    dGammadt = dsintheta_dtheta(a, theta, k)
    return dGammadt * dtdy
end

function dsintheta_dtheta(
    a :: SclavounosULLT,
    theta :: Real,
    k :: Int)
    
    assert(0 <= theta <= pi)
    dGamma_dt = (2 * k + 1) * cos((2 * k + 1) * theta)
    return dGamma_dt
end

function y_to_theta(
    a :: SclavounosULLT,
    y :: Real)

    assert(abs(y) < a.semispan)
    return acos(y / a.semispan)
end

function k_term1_singularity(
    a :: SclavounosULLT,
    delta_y :: Real)

    assert(delta_y != 0, "delta_y == 0 leads to NaN (inf) answer")
    return 1 / delta_y
end

function k_term1_numerator(
    a :: SclavounosULLT,
    delta_y :: Real)

    return exp(- (a.angular_fq / a.free_stream_vel) * abs(delta_y)) / 2
end

function k_term2(
    a :: SclavounosULLT,
    delta_y :: Real)

    coeff = sign(delta_y) / 2
    nu = a.angular_fq / a.free_stream_vel
    # We'd like to use the E_1 (exponential integral function) here, but
    # it isn't implemented in Julia. We'll use the identity  
    # E_n(x) = x^(n-1) Gamma(1-n, x)
    e1_term = - im * nu * SpecialFunctions.eint(nu * abs(delta_y))
    return coeff * -1 * e1_term
end

function k_term3(
    a :: SclavounosULLT,
    delta_y :: Real)

    coeff = - sign(delta_y) / 2
    nu = a.angular_fq / a.free_stream_vel
    p = nu * p(nu * abs(delta_y))

    return coeff * p
end

function p_eq(
    a :: SclavounosULLT,
    delta_y :: Real)

    function integrand1(t :: Float64)
        return - delta_y * exp(-delta_y * t) * (asin(1 / t) + sqrt(t^2 -1) -t)
    end
    function integrand2(t :: Float64)
        return exp(-delta_y * t) * (sqrt(1 - t^2) - 1) / t
    end

    term1 = -exp(-delta_y) * (pi/2 - 1) - QuadGK.quadgk(integrand1, 1, Inf)
    term2 = QuadGK.quadgk(integrand2, 0, 1)
    return term1 + im * term2
end

function integrate_gammaprime_k(
    a :: SclavounosULLT,
    y :: Real,
    k :: Int)

    assert(k >= 0)
    assert( abs(y) <= a.semispan )
    
    i1 = integrate_gammaprime_K_term1(a, y, k)
    i2 = integrate_gammaprime_K_term2(a, y, k)
    i3 = integrate_gammaprime_K_term3(a, y, k)
    
    return i1 + i2 + i3
end

function integrate_gammaprime_K_term1(
    a :: SclavounosULLT,
    y :: Real,
    k :: Int )
    
    theta_singular = y_to_theta(a, y)
    
    # We're using the singularity subtraction method to deal with a CPV problem.
    singularity_coefficient = 
        dsintheta_dtheta(theta_sing, k) * K_term1_numerator(a, 0)
    
    function integrand(theta_0 :: Real)
        eta = theta_to_y(a, theta_0)
        singular_part = K_term1_singularity(a, y - eta)
        nonsingular_K = K_term1_numerator(a, y - eta)
        gamma_dtheta = dsintheta_dtheta(a, theta0, k)
        
        singular_subtraction = nonsingular_K  * gamma_dtheta - 
                                                    singularity_coefficient
        return singular_part * singularity_subtraction
    end
    
    integral =
        QuadGK.quadgk(integrand, 0, theta_sing) +
        QuadGK.quadgk(integrand, theta_sing, pi) +
        singularity_coefficient * 0. # Glauert integral
    
    return -integral
end

function integrate_gammaprime_K_term2(
    a :: SclavounosULLT,
    y :: Real,
    k :: Int )
    
    assert(abs(y) < a.semispan)
    assert(k >= 0)
    theta_singular = y_to_theta(a, y)
    v = a.angular_fq / a.free_stream_vel
    
    integral_coefficient = im * v / 2
    function nonsingular_integrand(theta_0)
        return (2*k+1) * cos((2*k+1)*theta_0) / (v * a.semispan() * sin(theta))
    end
    function singular_integrand(theta_0)
        return v * a.semispan * sin(theta_0) * sign(theta_0 - theta_singular) * 
            SpecialFunctions.eint(v * a.semispan * abs(cos(theta_singular) - cos(theta_0)))
    end
    
    # The singular part (in terms of the singularity subtraction method) of the integral
    singular_integral = v * a.semispan * (
        (cos(theta_singular) + 1) * 
            SpecialFunctions.eint(v * a.semispan * (cos(theta_singular) + 1)) +
        (cos(theta_singular) - 1) * 
            SpecialFunctions.eint(v * a.semispan * (1 - cos(theta_sing)))) +
        exp( v * a.semispan * (cos(theta_singular) - 1)) -
        exp(-v * a.semispan * (cos(theta_singular) + 1))
        
    ssm_variable = non_singular(theta_singular)
    function numerical_integrand(theta_0)
        singular_var = singular_integrand(theta_0)
        non_singular_var = nonsingular_integrand(theta_0)
        singularity_subtraction = non_singular_var - ssm_variable
        integrand = singular_var * singularity_subtraction
        return integrand
    end
    
    int_lower = QuadGK.quadgk(numerical_integrand, 0, theta_singular)
    int_upper = QuadGK.quadgk(numerical_integrand, theta_singular, pi)
    complete_integral = integral_coefficient * (int_lower + int_upper + ssm_variable * singular_integral)
    return complete_integral
end

function integrate_gammaprime_K_term3(
    a :: SclavounosULLT,
    y :: Real,
    k :: Int)
        
    theta_singular = y_to_theta(a, y)
    function integrand(theta_0)
        eta = theta_to_y(a, theta_0)
        return dsintheta_dtheta(a, theta_0, k) * K_term3(y - eta)
    end
    
    integral =  QuadGK.quadgk(integrand, 0, theta_singular)  +
                QuadGK.quadgk(integrand, theta_singular, pi)
    return integral
end

function F(
    a :: SclavounosULLT,
    y :: Real )
    
    assert(abs(y) <= a.semispan)
    error("Managing j=3/5")
end

function gamma_terms_matrix(
    a :: SclavounosULLT )

    idxs = collect(0:a.num_terms)
    mtrx = map(
        (i, j)->sin((2 * j + 1) * a.collocation_points[i]),
        collect((i,j) for i in idxs, j in idxs)
    )
    return mtrx
end

function rhs_vector(
    a :: SclavounosULLT )

    assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "Sclavounos.jl: SclavounosULLT.pitch_plunge must be 3" *
        " (plunge) or 5 (pitch). Value was ", a.pitch_plunge, "." )
    if(a.pitch_plunge == 3) # Plunge
        circ_vect = map(
            theta->d3(a, theta_to_y(a, theta)),
            a.collocation_points
        )
    elif(a.pitch_plunge == 5)   # Pitch
        circ_vect = map(
            theta->d5(a, theta_to_y(a, theta)) 
                - a.free_stream_vel * d3(a, theta_to_y(a, theta))/ (im * a.angular_fq),
            a.collocation_points
        )
    end
    return circ_vect
end

function compute_solution(
    a :: SclavounosULLT )

    gamma_mtrx = gamma_terms_matrix(a)
    integro_diff_mtrx = 
        map(
            (i, j)->
                integro_diff_coeff(a, i) * 
                integrate_gammaprime_k(a,  y_to_theta(a, a.collocation_points[i], j)),
            collect((i, j) for i in 0:a.num_terms-1, j in 0:a.num_terms-1)
        )
    rhs_vec = rhs_vector(a)
    solution = (gamma_mtrx - integro_diff_mtrx) \ rhs_vec
    a.solution = solution
    return
end    

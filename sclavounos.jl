import SpecialFunctions
import FastGaussQuadrature
import HCubature

@enum DownwashModel begin
    psuedosteady = 1
    extpsuedosteady = 2
    unsteady = 3
    strip_theory = 4
end

mutable struct StraightAnalyticWing
    semispan :: Real        # Half the full span of the wing
    chord_fn :: Function    # Defined in [-semispan, semispan]

    function StraightAnalyticWing(
        semispan :: Real, chord_fn :: Function) 
        new(semispan, chord_fn)
    end
end

function make_rectangular(
    ::Type{StraightAnalyticWing}, 
    aspect_ratio :: Real, span :: Real ) 

    fn = y -> span / aspect_ratio
    semispan = span / 2
    return StraightAnalyticWing(semispan, fn)
end

function make_elliptic(
    ::Type{StraightAnalyticWing}, 
    aspect_ratio :: Real, span :: Real )

    semispan = span / 2
    fn = y -> (4 * semispan/ (aspect_ratio * pi)) * sqrt(semispan^2 - y^2)
    return StraightAnalyticWing(semispan, fn)
end

function make_van_dyke_cusped(
    ::Type{StraightAnalyticWing},
    aspect_ratio :: Real, span :: Real, n :: Int)

    @assert(n >= 0, "N must be postive")
    @assert(n < 7, "Only implemented for n < 7")
    semispan = span / 2
    kns = [1, 4/pi, 3/2, 16/(3 * pi), 15/8, 32/(5*pi), 32/16, 256/(35*pi)]
    kn = kns[n + 1]
    fn = y->kn * (1 - y^2 / semispan^2)^(n/2)
    return StraightAnalyticWing(semispan, fn)
end

mutable struct HarmonicULLT
    angular_fq :: Real              # in rad / s
    free_stream_vel :: Real

    wing :: StraightAnalyticWing
    amplitude_fn :: Function        # Currently unused.
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

"""
    d3(::HarmonicULLT, ::Real)

Computes the normalised bound circulation on a harmonically plunging plate.
"""
function d3(
    a :: HarmonicULLT,
    y :: Real)

    @assert(abs(y) <= a.wing.semispan)
    norm_fq = a.angular_fq / a.free_stream_vel
    semichord = a.wing.chord_fn(y) / 2
    num = 4 * a.free_stream_vel * exp(-im * semichord * norm_fq)
    den = im * SpecialFunctions.hankelh2(0, norm_fq * semichord) +
        SpecialFunctions.hankelh2(1, norm_fq * semichord)
    return num / den
end

"""
    d5(::HarmonicULLT, ::Real)

Computes the normalised bound circulation on a harmonically pitching plate.
"""
function d5(
    a :: HarmonicULLT,
    y :: Real)

    chord = a.wing.chord_fn(y)
    return -chord * sclavounos_d3(HarmonicULLT, y) / 4
end

"""
    compute_collocation_points!(::HarmonicULLT)

Computes the correct location of collocation points for the fourier
approximation and applies to input HarmonicULLT
"""
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

"""
    theta_to_y(::HarmonicULLT, theta::Real)

Converts angular location on wing span in [0, pi] to the global y 
location in [-semispan, semispan]
"""
function theta_to_y(
    a :: HarmonicULLT,
    theta :: Real)

    @assert(0 <= theta <= pi)
    return a.wing.semispan * cos(theta)
end

"""
    dtheta_dy(::HarmonicULLT, theta::Real)

Computes the rate of change of theta with respect to global y.
"""
function dtheta_dy(
    a :: HarmonicULLT,
    y :: Real)
    
    @assert(abs(y) <= a.wing.semispan)
    result = -1 / sqrt(pow(a.wing.semispan, 2) - y^2)
    return result
end

function dsintheta_dy(
    a :: HarmonicULLT,
    y :: Real,
    k :: Integer)
    
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
    
    @assert(k >= 0, "Positive wavenumbers only please!")
    @assert(0 <= theta <= pi)
    dGamma_dt = (2 * k + 1) * cos((2 * k + 1) * theta)
    return dGamma_dt
end

"""
    y_to_theta(::HarmonicULLT, y::Real)

Converts the global y location in [-semispan, semispan] to the angular location 
on wing span in [0, pi].
"""
function y_to_theta(
    a :: HarmonicULLT,
    y :: Real)

    @assert(abs(y) <= a.wing.semispan)
    return acos(y / a.wing.semispan)
end

function k_term1_singularity(
    a :: HarmonicULLT,
    delta_y :: Real) 

    @assert(delta_y != 0, "delta_y == 0 leads to NaN (inf) answer")
    return 1 / delta_y
end

function k_term1_numerator(
    a :: HarmonicULLT,
    delta_y :: Real)

    return exp(- (a.angular_fq / a.free_stream_vel) * abs(delta_y)) / 2
end

function k_term2(
    a :: HarmonicULLT,
    delta_y :: Real)

    coeff = sign(delta_y) / 2
    nu = a.angular_fq / a.free_stream_vel
    e1_term = - im * nu * expint(nu * abs(delta_y))
    return coeff * -1 * e1_term
end

function k_term3(
    a :: HarmonicULLT,
    delta_y :: Real)

    coeff = - sign(delta_y) / 2
    nu = a.angular_fq / a.free_stream_vel
    p = nu * p_eq(a, nu * abs(delta_y))

    return coeff * p
end

function p_eq(
    a :: HarmonicULLT,
    delta_y :: Real)

    function integrand1(t :: T) where T <: Real
        val = - delta_y * exp(-delta_y * t) * (asin(1 / t) + sqrt(t^2 -1) -t)
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

function integrate_gammaprime_k(
    a :: HarmonicULLT,
    y :: Real,
    k :: Integer)

    @assert(k >= 0)
    @assert( abs(y) <= a.wing.semispan )
    
    if( a.downwash_model == unsteady )
        i1 = integrate_gammaprime_k_term1(a, y, k)
        i2 = integrate_gammaprime_k_term2(a, y, k)
        i3 = integrate_gammaprime_k_term3(a, y, k)
        integral = i1 + i2 + i3
    elseif( a.downwash_model == psuedosteady )
        integral = integrate_gammaprime_k_psuedosteady(a, y, k)
    elseif( a.downwash_model == extpsuedosteady )
        integral = integrate_gammaprime_k_ext_psuedosteady(a, y, k)
    elseif( a.downwash_model == strip_theory )
        integral = 0
    end
    return integral
end

function integrate_gammaprime_k_term1(
    a :: HarmonicULLT,
    y :: Real,
    k :: Integer)
    
    theta_singular = y_to_theta(a, y)
    
    # We're using the singularity subtraction method to deal with a CPV problem.
    singularity_coefficient = 
        dsintheta_dtheta(a, theta_singular, k) * k_term1_numerator(a, 0)
    
    function integrand(theta_0 :: Real)
        eta = theta_to_y(a, theta_0)
        singular_part = k_term1_singularity(a, y - eta)
        nonsingular_K = k_term1_numerator(a, y - eta)
        gamma_dtheta = dsintheta_dtheta(a, theta_0, k)
        
        singular_subtraction = nonsingular_K  * gamma_dtheta - 
                                                    singularity_coefficient
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

function integrate_gammaprime_k_term2(
    a :: HarmonicULLT,
    y :: Real,
    k :: Integer)
    
    @assert(abs(y) < a.wing.semispan)
    @assert(k >= 0)
    theta_singular = y_to_theta(a, y)
    v = a.angular_fq / a.free_stream_vel
    
    integral_coefficient = im * v / 2
    function nonsingular_integrand(theta_0)
        return (2*k+1) * cos((2*k+1)*theta_0) / (v * a.wing.semispan * sin(theta_0))
    end
    function singular_integrand(theta_0)
        return v * a.wing.semispan * sin(theta_0) * sign(theta_0 - theta_singular) * 
            expint(v * a.wing.semispan * abs(cos(theta_singular) - cos(theta_0)))
    end
    
    # The singular part (in terms of the singularity subtraction method) of the integral
    singular_integral = v * a.wing.semispan * (
        (cos(theta_singular) + 1) * 
            expint(v * a.wing.semispan * (cos(theta_singular) + 1)) +
        (cos(theta_singular) - 1) * 
            expint(v * a.wing.semispan * (1 - cos(theta_singular)))) +
        exp( v * a.wing.semispan * (cos(theta_singular) - 1)) -
        exp(-v * a.wing.semispan * (cos(theta_singular) + 1))
        
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
    complete_integral = integral_coefficient * (int_lower + int_upper + ssm_variable * singular_integral)
    return complete_integral
end

function integrate_gammaprime_k_term3(
    a :: HarmonicULLT,
    y :: Real,
    k :: Integer)
        
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
    return integral
end

function integrate_gammaprime_k_psuedosteady(
    a :: HarmonicULLT, 
    y :: Real, 
    k :: Integer)

    theta = y_to_theta(a, y)
    integral = (2*k + 1) * pi * sin((2*k + 1) * theta) / 
        (2 * a.wing.semispan * sin(theta))
    return integral
end

function integrate_gammaprime_k_ext_psuedosteady(
    a :: HarmonicULLT,
    y :: Real,
    k :: Integer)

    theta_singular = y_to_theta(a, y)
    ssm_var = 1 # save evaluating 
        # integrate_gammaprime_k_ext_psuedosteady_subint(a, 0, k)
    function integrand(theta_0)
        eta = theta_to_y(a, theta_0)
        singular = a.wing.semispan * cos((2*k +1)*theta_0) / (y - eta)
        non_singular = 
            integrate_gammaprime_k_ext_psuedosteady_subint(a, y - eta, k)
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

function integrate_gammaprime_k_ext_psuedosteady_subint(
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

function gamma_terms_matrix(
    a :: HarmonicULLT )
    idxs = collect(1:a.num_terms)
    mtrx = map(
        i->sin((2 * i[2] - 1) * a.collocation_points[i[1]]),
        collect((i,j) for i in idxs, j in idxs)
    )
    return mtrx
end

function rhs_vector(
    a :: HarmonicULLT )

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "Sclavounos.jl: HarmonicULLT.pitch_plunge must be 3" *
        " (plunge) or 5 (pitch). Value was ", a.pitch_plunge, "." )
    if(a.pitch_plunge == 3) # Plunge
        circ_vect = map(
            theta->d3(a, theta_to_y(a, theta)),
            a.collocation_points
        )
    elseif(a.pitch_plunge == 5)   # Pitch
        circ_vect = map(
            theta->d5(a, theta_to_y(a, theta)) 
                - a.free_stream_vel * d3(a, theta_to_y(a, theta))/ (im * a.angular_fq),
            a.collocation_points
        )
    end
    return circ_vect
end

function integro_diff_mtrx_coeff(
    a :: HarmonicULLT,
    y_pos :: Real)

    coeff = d3(a, y_pos) / (2 * pi * a.angular_fq * im)
    return coeff
end

"""
    compute_fourier_terms!(::HarmonicULLT)

Computes the fourier terms representing the bound vorticity on the span of the 
wing. Assumes collocation points have already been computed.
"""
function compute_fourier_terms!(
    a :: HarmonicULLT )

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

"""
    compute_lift_coefficient(::HarmonicULLT)

Compute the lift coefficient on a solved HarmonicULLT.

# Example
'''julia-repl
julia> prob = HarmonicULLT(
    1,  # Frequency
    1,  # Free stream vel
    2,  # Semispan
    y->1,   # Chord with respect to span (rectangular)
    y->1,   # Displacement with repect to span
    3,  # Plunge (3) or pitch (5).
    unsteady, 
    8,  # Number of fourier terms
    Vector{Float64}([]), # Vector/type for collocation points
    Vector{Float64}([])) # Vector/type for solution
julia> compute_collocation_points(prob)
julia> compute_fourier_terms!(prob)
julia> compute_lift_coefficient(prob)
'''
"""
function compute_lift_coefficient(
    a :: HarmonicULLT )

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "HarmonicULLT.pitch_plunge must equal 3 (plunge) or 5 (pitch).")

    w_area = wing_area(a)
    if(a.pitch_plunge == 3)   # Plunge
        function circulatory_integrand(theta :: T) where T <: Real
            y = theta_to_y(a, theta)
            semichord = a.wing.chord_fn(y) / 2
            f_3 = f_eq(a, y)
            c = theodorsen_fn((a.angular_fq / a.free_stream_vel) * semichord)
            return c * semichord * pi * (1 - f_3) * sin(theta)
        end
        circulatory_integral_coeff = -8 * a.wing.semispan / w_area
        circulatory_integral = 
            HCubature.hquadrature(circulatory_integrand, 0, pi/2)[1] # Assumes symettric
        circulatory_part = circulatory_integral_coeff * circulatory_integral

        added_mass_coeff = -1 * im * (a.angular_fq / a.free_stream_vel)
        added_mass_integral = wing_heave_added_mass(a) / w_area
        added_mass_part = added_mass_coeff *  added_mass_integral
        cl_val = circulatory_part + added_mass_part
    elseif(a.pitch_plunge == 5)  # Pitch
        function integrand(theta :: T) where T <: Real
            y = theta_to_y(theta :: T) where T <: Real
            semichord = a.wing.chord_fn(y) / 2
            f_5 = f_eq(a, y)
            c = theodorsen_fn((a.angular_fq / a.free_stream_vel) * semichord)
            circ = semichord + 
                (2 * a.free_stream_vel / (im * a.angular_fq) + 2 * f_5)
            added_mass = semichord * 
                (1 + im * a.angular_fq * f_5 / a.free_stream_vel)
            return pi * semichord * sin(theta) * (c * circ + added_mass)
        end
        coeff = 4 * a.wing.semispan / w_area
        integral = HCubature.hquadrature(integrand, 0, pi/2)[1] # Assume symettric
        cl_val = coeff * integral
    end
    return cl_val
end

function f_eq(
    a :: HarmonicULLT,
    y :: Real)

    @assert(abs(y) <= a.wing.semispan)
    if (a.pitch_plunge == 3)
        f = 1 - bound_vorticity(a, y) / d3(a, y)
    else    # a.pitch_plunge == 5 
        f = (d5(a, y) - bound_vorticity(a, y)) / d3(a, y) - 
            a.free_stream_vel / (im * a.angular_fq)
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

"""
    theodorsen_fn(k::Real)

Theodorsen's function C(k), where k is chord reduced frequency = omega c / 2 U.
"""
function theodorsen_fn(k :: Real)
    @assert(k >= 0, "Chord reduced frequency should be positive.")

    h21 = SpecialFunctions.hankelh2(1, k)
    h20 = SpecialFunctions.hankelh2(0, k)
    return h21 / (h21 + im * h20)
end

function wing_area(
    a :: HarmonicULLT)

    # Integrate chord from accross span.
    return HCubature.hquadrature(a.wing.chord_fn, -a.wing.semispan, a.wing.semispan)[1]
end

function wing_heave_added_mass(
    a :: HarmonicULLT ) 

    function integrand(y :: T) where T <: Real
        semichord = a.wing.chord_fn(y) / 2
        return semichord * semichord
    end
    integral = HCubature.hquadrature(integrand, -a.wing.semispan, a.wing.semispan)[1]
    return 2 * pi * integral
end

function linear_remap(
    pointin :: Number,   weightin :: Number,
    old_a :: Number,     old_b :: Number,
    new_a :: Number,     new_b :: Number )

    dorig = (pointin - old_a) / (old_b - old_a)
    p_new = new_a + dorig * (new_b - new_a)
    w_new = ((new_b - new_a) / (old_b - old_a)) * weightin
    return p_new, w_new
end

#==============================================================================#
#   EXPONENTIAL INTEGRAL                                                      
#   Julia does not have a exponential integral implementation. This is 
#   copied from 
#   https://github.com/mschauer/Bridge.jl/blob/master/src/expint.jl            
#   under MIT lisense. Credit to stevengj and mschauer.                                    

using Base.MathConstants: eulergamma

# n coefficients of the Taylor series of E₁(z) + log(z), in type T:
function E₁_taylor_coefficients(::Type{T}, n::Integer) where T<:Number
    n < 0 && throw(ArgumentError("$n ≥ 0 is required"))
    n == 0 && return T[]
    n == 1 && return T[-eulergamma]
    # iteratively compute the terms in the series, starting with k=1
    term::T = 1
    terms = T[-eulergamma, term]
    for k in 2:n
        term = -term * (k-1) / (k * k)
        push!(terms, term)
    end
    return terms
end

# inline the Taylor expansion for a given order n, in double precision
macro E₁_taylor64(z, n::Integer)
    c = E₁_taylor_coefficients(Float64, n)
    taylor = :(@evalpoly zz)
    append!(taylor.args, c)
    quote
        let zz = $(esc(z))
            $taylor - log(zz)
        end
    end
end

# for numeric-literal coefficients: simplify to a ratio of two polynomials:
import Polynomials
# return (p,q): the polynomials p(x) / q(x) corresponding to E₁_cf(x, a...),
# but without the exp(-x) term
function E₁_cfpoly(n::Integer, ::Type{T}=BigInt) where T<:Real
    q = Polynomials.Poly(T[1])
    p = x = Polynomials.Poly(T[0,1])
    for i in n:-1:1
        p, q = x*p+(1+i)*q, p # from cf = x + (1+i)/cf = x + (1+i)*q/p
        p, q = p + i*q, p     # from cf = 1 + i/cf = 1 + i*q/p
    end
    # do final 1/(x + inv(cf)) = 1/(x + q/p) = p/(x*p + q)
    return p, x*p + q
end
macro E₁_cf64(z, n::Integer)
    p, q = E₁_cfpoly(n, BigInt)
    num_expr =  :(@evalpoly zz)
    append!(num_expr.args, Float64.(Polynomials.coeffs(p)))
    den_expr = :(@evalpoly zz)
    append!(den_expr.args, Float64.(Polynomials.coeffs(q)))
    quote
        let zz = $(esc(z))
            exp(-zz) * $num_expr / $den_expr
        end
    end
end

# exponential integral function E₁(z)
function expint(z::Union{Float64,Complex{Float64}})
    x² = real(z)^2
    y² = imag(z)^2
    if real(z) > 0 && x² + 0.233*y² ≥ 7.84 # use cf expansion, ≤ 30 terms
        if (x² ≥ 546121) & (real(z) > 0) # underflow
            return zero(z)
        elseif x² + 0.401*y² ≥ 58.0 # ≤ 15 terms
            if x² + 0.649*y² ≥ 540.0 # ≤ 8 terms
                x² + y² ≥ 4e4 && return @E₁_cf64 z 4
                return @E₁_cf64 z 8
            end
            return @E₁_cf64 z 15
        end
        return @E₁_cf64 z 30
    else # use Taylor expansion, ≤ 37 terms
        r² = x² + y²
        return r² ≤ 0.36 ? (r² ≤ 2.8e-3 ? (r² ≤ 2e-7 ? @E₁_taylor64(z,4) :
                                                       @E₁_taylor64(z,8)) :
                                                       @E₁_taylor64(z,15)) :
                                                       @E₁_taylor64(z,37)
    end
end

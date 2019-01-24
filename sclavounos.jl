import SpecialFunctions
import FastGaussQuadrature
import HCubature

@enum DownwashModel begin
    psuedosteady = 1
    extpsuedosteady = 2
    unsteady = 3
end

mutable struct SclavounosULLT
    angular_fq :: Real              # in rad / s
    free_stream_vel :: Real
    semispan :: Real

    chord_fn :: Function            # Defined for [-semispan, semispan]
    amplitude_fn :: Function        # Currently unused.
    pitch_plunge :: Int64           # Plunge = 3, Pitch = 5. Otherwise invalid.

    downwash_model :: DownwashModel # See DownwashModel defintion
    num_terms :: Int64              # Number of terms in fourier expansion
    fourier_terms :: Vector{Complex{Real}}
    collocation_points :: Vector{Real}  # In terms of theta in [0, pi]
end

function d3(
    a :: SclavounosULLT,
    y :: T) where T <: Real

    @assert(abs(y) <= a.semispan)
    norm_fq = a.angular_fq / a.free_stream_vel
    semichord = a.chord_fn(y) / 2
    num = 4 * a.free_stream_vel * exp(-im * semichord * norm_fq)
    den = im * SpecialFunctions.hankelh2(0, norm_fq * semichord) +
        SpecialFunctions.hankelh2(1, norm_fq * semichord)
    return num / den
end

function d5(
    a :: SclavounosULLT,
    y :: T) where T <: Real

    chord = a.chord_fn(y)
    return -chord * sclavounos_d3(SclavounosULLT, y) / 4
end

function compute_collocation_points(
    a :: SclavounosULLT)

    nt = a.num_terms
    @assert(nt > 0, "Number of terms in SclavounosULLT must be more than 0")
    pos = Vector{Float64}(undef, a.num_terms)
    hpi = pi / 2
    for i = 1 : nt
        pos[i] = (pi * i - hpi) / (2 * nt)
    end
    a.collocation_points = pos
    return
end

function theta_to_y(
    a :: SclavounosULLT,
    theta :: T) where T <: Real

    @assert(0 <= theta <= pi)
    return a.semispan * cos(theta)
end

function dtheta_dy(
    a :: SclavounosULLT,
    y :: T) where T <: Real
    
    @assert(abs(y) <= a.semispan)
    result = -1 / sqrt(pow(a.semispan, 2) - y^2)
    return result
end

function dsintheta_dy(
    a :: SclavounosULLT,
    y :: T,
    k :: S) where {T <: Real, S <: Integer}
    
    theta = y_to_theta(a, y)
    dtdy = dtheta_dy(y)
    dGammadt = dsintheta_dtheta(a, theta, k)
    return dGammadt * dtdy
end

function dsintheta_dtheta(
    a :: SclavounosULLT,
    theta :: T,
    k :: S) where {T <: Real, S <: Integer}
    
    @assert(0 <= theta <= pi)
    dGamma_dt = (2 * k + 1) * cos((2 * k + 1) * theta)
    return dGamma_dt
end

function y_to_theta(
    a :: SclavounosULLT,
    y :: T) where {T <: Real}

    @assert(abs(y) <= a.semispan)
    return acos(y / a.semispan)
end

function k_term1_singularity(
    a :: SclavounosULLT,
    delta_y :: T) where {T <: Real}

    @assert(delta_y != 0, "delta_y == 0 leads to NaN (inf) answer")
    return 1 / delta_y
end

function k_term1_numerator(
    a :: SclavounosULLT,
    delta_y :: T) where {T <: Real}

    return exp(- (a.angular_fq / a.free_stream_vel) * abs(delta_y)) / 2
end

function k_term2(
    a :: SclavounosULLT,
    delta_y :: T) where {T <: Real}

    coeff = sign(delta_y) / 2
    nu = a.angular_fq / a.free_stream_vel
    e1_term = - im * nu * expint(nu * abs(delta_y))
    return coeff * -1 * e1_term
end

function k_term3(
    a :: SclavounosULLT,
    delta_y :: T) where {T <: Real}

    coeff = - sign(delta_y) / 2
    nu = a.angular_fq / a.free_stream_vel
    p = nu * p_eq(a, nu * abs(delta_y))

    return coeff * p
end

function p_eq(
    a :: SclavounosULLT,
    delta_y :: T) where {T <: Real}

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
    a :: SclavounosULLT,
    y :: T,
    k :: S) where {T <: Real, S <: Integer}

    @assert(k >= 0)
    @assert( abs(y) <= a.semispan )
    
    i1 = integrate_gammaprime_k_term1(a, y, k)
    i2 = integrate_gammaprime_k_term2(a, y, k)
    i3 = integrate_gammaprime_k_term3(a, y, k)
    #println("y = ", y, ", k = ", k)
    #println("i1 = ", i1, "\ti2 = ", i2, "\ti3 = ", i3)
    return i1 + i2 + i3
end

function integrate_gammaprime_k_term1(
    a :: SclavounosULLT,
    y :: T,
    k :: S) where {T <: Real, S <: Integer}
    
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
    a :: SclavounosULLT,
    y :: T,
    k :: S) where {T <: Real, S <: Integer}
    
    @assert(abs(y) < a.semispan)
    @assert(k >= 0)
    theta_singular = y_to_theta(a, y)
    v = a.angular_fq / a.free_stream_vel
    
    integral_coefficient = im * v / 2
    function nonsingular_integrand(theta_0)
        return (2*k+1) * cos((2*k+1)*theta_0) / (v * a.semispan * sin(theta_0))
    end
    function singular_integrand(theta_0)
        return v * a.semispan * sin(theta_0) * sign(theta_0 - theta_singular) * 
            expint(v * a.semispan * abs(cos(theta_singular) - cos(theta_0)))
    end
    
    # The singular part (in terms of the singularity subtraction method) of the integral
    singular_integral = v * a.semispan * (
        (cos(theta_singular) + 1) * 
            expint(v * a.semispan * (cos(theta_singular) + 1)) +
        (cos(theta_singular) - 1) * 
            expint(v * a.semispan * (1 - cos(theta_singular)))) +
        exp( v * a.semispan * (cos(theta_singular) - 1)) -
        exp(-v * a.semispan * (cos(theta_singular) + 1))
        
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
    a :: SclavounosULLT,
    y :: T,
    k :: S) where {T <: Real, S <: Integer}
        
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

function gamma_terms_matrix(
    a :: SclavounosULLT )
    idxs = collect(1:a.num_terms)
    mtrx = map(
        i->sin((2 * i[2] - 1) * a.collocation_points[i[1]]),
        collect((i,j) for i in idxs, j in idxs)
    )
    return mtrx
end

function rhs_vector(
    a :: SclavounosULLT )

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "Sclavounos.jl: SclavounosULLT.pitch_plunge must be 3" *
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
    a :: SclavounosULLT,
    y_pos :: T) where T <: Real

    coeff = d3(a, y_pos) / (2 * pi * a.angular_fq * im)
    return coeff
end

function compute_fourier_terms(
    a :: SclavounosULLT )

    #println("In compute fourier terms")
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

function compute_lift_coefficient(
    a :: SclavounosULLT )

    @assert((a.pitch_plunge == 3) || (a.pitch_plunge == 5),
        "SclavounosULLT.pitch_plunge must equal 3 (plunge) or 5 (pitch).")

    w_area = wing_area(a)
    if(a.pitch_plunge == 3)   # Plunge
        function circulatory_integrand(theta :: T) where T <: Real
            y = theta_to_y(a, theta)
            semichord = a.chord_fn(y) / 2
            f_3 = f_eq(a, y)
            c = theodorsen_fn((a.angular_fq / a.free_stream_vel) * semichord)
            return c * semichord * pi * (1 - f_3) * sin(theta)
        end
        circulatory_integral_coeff = -8 * a.semispan / w_area
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
            semichord = a.chord_fn(y) / 2
            f_5 = f_eq(a, y)
            c = theodorsen_fn((a.angular_fq / a.free_stream_vel) * semichord)
            circ = semichord + 
                (2 * a.free_stream_vel / (im * a.angular_fq) + 2 * f_5)
            added_mass = semichord * 
                (1 + im * a.angular_fq * f_5 / a.free_stream_vel)
            return pi * semichord * sin(theta) * (c * circ + added_mass)
        end
        coeff = 4 * a.semispan / w_area
        integral = HCubature.hquadrature(integrand, 0, pi/2)[1] # Assume symettric
        cl_val = coeff * integral
    end
    return cl_val
end

function f_eq(
    a :: SclavounosULLT,
    y :: T) where T <: Real

    @assert(abs(y) <= a.semispan)
    if (a.pitch_plunge == 3)
        f = 1 - bound_vorticity(a, y) / d3(a, y)
    else    # a.pitch_plunge == 5 
        f = (d5(a, y) - bound_vorticity(a, y)) / d3(a, y) - 
            a.free_stream_vel / (im * a.angular_fq)
    end
    return f
end

function bound_vorticity(
    a :: SclavounosULLT,
    y :: T) where T <: Real

    @assert(abs(y) <= a.semispan)
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

function theodorsen_fn(k :: T) where T <: Real
    @assert(k >= 0, "Chord reduced frequency should be positive.")

    h21 = SpecialFunctions.hankelh2(1, k)
    h20 = SpecialFunctions.hankelh2(0, k)
    return h21 / (h21 + im * h20)
end

function wing_area(
    a :: SclavounosULLT)

    # Integrate chord from accross span.
    return HCubature.hquadrature(a.chord_fn, -a.semispan, a.semispan)[1]
end

function wing_heave_added_mass(
    a :: SclavounosULLT ) 

    function integrand(y :: T) where T <: Real
        semichord = a.chord_fn(y) / 2
        return semichord * semichord
    end
    integral = HCubature.hquadrature(integrand, -a.semispan, a.semispan)[1]
    return 2 * pi * integral
end

function linear_remap(
    pointin :: T,   weightin :: T,
    old_a :: R,     old_b :: S,
    new_a :: Q,     new_b :: U ) where {
        T <: Number, w <: Number, 
        R <: Number, S <: Number, 
        Q <: Number, U <: Number}

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

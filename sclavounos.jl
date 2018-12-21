
import SpecialFunctions

@enum DownwashModel 
    psuedosteady = 1
    extpsuedosteady = 2
    unsteady = 3
end


mutable struct SclavounosULLT
    angular_fq :: Float64
    free_stream_vel :: Float64
    semispan :: Float64

    chord_at_points :: Function
    pitch_location :: Float64
    pitch_amplitude_fn :: Function
    plunge_amplitude_fn :: Function

    downwash_model :: Function
    num_terms :: In64
    terms :: Vector{Complex{Float64}}
    collocation_points :: Vector{Float64}
end

function d3(
    a :: SclavounosULLT,
    chord :: Float64)

    norm_fq = a.angular_fq / a.free_stream_vel
    semichord = chord / 2
    num = 4 * U * exp(-im * semichord * norm_fq)
    den = im * SpecialFunctions.hankelh2(0, norm_fq * semichord) +
        SpecialFunctions.hankelh2(1, norm_fq * semichord)
    return num / den
end

function d5(
    a :: SclavounosULLT,
    chord :: Float64)

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

    
end



mutable struct GuermondUnsteady2
    angular_fq :: Real              # in rad / s
    free_stream_vel :: Real

    wing :: StraightAnalyticWing
    amplitude_fn :: Function        # Amplitude of oscillation wrt/ span pos.
    pitch_plunge :: Int64           # Plunge = 3, Pitch = 5. Otherwise invalid.

    chord_len_scale :: Float64
    span_len_scale :: Float64
    guermond_k :: Float64
    guermond_nu :: Float64

    num_terms :: Int64
    fourier_terms :: Vector{ComplexF64}
    collocation_points :: Vector{Float64} # Collocation points in [0, pi]

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


function Y_to_y(a::GuermondUnsteady2, Y::Real)
    return Y / a.wing.semispan
end

function Y_to_y(a::GuermondUnsteady2, Y::Vector{<:Real})
    return Y ./ a.wing.semispan
end

function y_to_Y(a::GuermondUnsteady2, y::Real)
    return y * a.wing.semispan
end

function y_to_Y(a::GuermondUnsteady2, y::Vector{<:Real})
    return y .* a.wing.semispan
end

function compute_collocation_points!(
    a :: GuermondUnsteady2)

    nt = a.num_terms
    @assert(nt > 0, "Number of terms in GuermondUnsteady2 must be more than 0")
    pos = Vector{Float64}(undef, a.num_terms)
    hpi = pi / 2
    for i = 1 : nt
        pos[i] = (pi * i - hpi) / (2 * nt)
    end
    a.collocation_points = pos
    return
end

function compute_fourier_terms!(
    a :: GuermondUnsteady2)

    terms2d = zeros(ComplexF64, a.num_terms)
    lhs_terms = zeros(ComplexF64, a.num_terms, a.num_terms)
    terms3d = zeros(ComplexF64, a.num_terms, a.num_terms)
    
    terms2d = g2d(a)


    a_terms = (lhs_terms .+ terms3d ./ aspect_ratio(a.wing)) \ terms2d
    a.fourier_terms = a.terms
    return
end

function g2d(a :: GuermondUnsteady2)
    ys = cos.(a.collocation_points)
    Ys = a.wing.semispan .* ys
    gs = Vector{ComplexF64}
    U = a.free_stream_vel
    guerk = a.guermond_k

    for i = 1 : length(ys)
        chord = a.wing.chord_fn(Ys[i])
        k = a.angular_fq * chord / (2* a.free_stream_vel)
        amp = a.amplitude_fn(Ys[i])
        if a.pitch_plunge == 3
            uw = make_plunge_function(HarmonicUpwash2D, amp, k;
                free_stream_vel = a.free_stream_vel, semichord=chord/2)
        elseif a.pitch_plunge == 5
            uw = make_pitch_function(HarmonicUpwash2D, amp, k;
                free_stream_vel = a.free_stream_vel, semichord=chord/2)
        end
        bv = bound_vorticity(uw)
        gs[i] = bv * exp(im * guerk * chord / 2)
    end
    return gs
end

function create_lhs_terms(a :: GuermondUnsteady2)
    t = zeros(ComplexF64, a.num_terms, a.num_terms)
    for i = 1 : a.num_terms
        for j = 1 : a.num_terms
            t[i, j] = sin(j * a.collocation_points[i])
        end
    end
    return t
end

function g3d(a::GuermondUnsteady2)
    t = zeros(ComplexF64, a.num_terms, a.num_terms)
    for i = 1 : a.num_terms
        for j = 1 : a.num_terms
            t = g3d_term(a, j, a.collocation_points[i])
        end
    end
end

function g3d_term(a::GuermondUnsteady2, 
    n :: Integer, theta :: Real)

    y = cos(theta)

    # The non-oscillatory term.
    f1 = eta -> sin(n * acos(eta)) / sqrt(1 - eta^2)
    function f1p(eta)
        ti1n = n * cos(n * acos(eta))
        ti1d = eta^2 - 1
        ti2n = eta * sin(n * acos(eta))
        ti2d = (1-eta^2)^(3/2)
        return ti1n / ti1d + ti2n / ti2d
    end
    nonosclterm = integrate_finite_part_chebyshev2(f1, f1p, y)

    # The inner integral of the oscillatory term
    function inner_integral_fx(eta, v)
        ti1 = sin(n * acos(eta)) / sqrt(1-eta^2)
        ti2 = 1 + v / sqrt(v^2 + (y - eta)^2)
        return ti1 * ti2
    end
    function inner_integral_fxp(eta, v)
        tia = 1 + v / sqrt(v^2 + (y - eta)^2)
        tib = sin(n * acos(eta))
        tic = eta^2 - 1

        ti1 = n * tia * cos(n * acos(eta)) / (eta^2 - 1)
        ti2 = eta * tia * tib / (1-eta^2)^(3/2)
        ti3 = v * (y-eta) * tib / (sqrt(1-eta^2) * (v^2 + (eta-y)^2)^(3/2))
        return ti1 + ti2 + ti3
    end
    function inner_terms(v)
        ti1 = integrate_finite_part_chebyshev2(
                x->inner_integral_fx(x, v), 
                x->inner_integral_fxp(x, v), y)
        ti2 = 2 / v # This is the term not in w_0(M)
        return ti1 + ti2
    end
    tosccl = fejer_quadrature_exp(inner_terms, a.guermond_nu, 
        -Inf, -1e-8; finite=true)
    total_term = nonosclterm / (4 * pi) - im * a.guermond_nu * tosccl / (4 * pi)
    
    chord = a.wing.chord_fn(y * a.wing.semispan)
    uw = make_sinusoidal_gust_function(HarmonicUpwash2D,
        total_term, a.guermond_k / 2; free_stream_vel= a.free_stream_vel, 
        semichord= chord / 2 )
    #uws = make_plunge_function(HarmonicUpwash2D, pd_coeff / a.angular_fq, k;
    #    free_stream_vel = a.free_stream_vel, semichord=chord/2)
    bv = bound_vorticity(uw)
    gs = bv * exp(im * a.guermond_k * chord / 2)
    return gs
end



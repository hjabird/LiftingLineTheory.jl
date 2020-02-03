#
# GuermondUnsteady2.jl
#
# A indirect method of solving Guermond & Sellier's 1991 ULLT similar to 
# Prandtl's method (insofar that it matches a fourier series in the inner and
# outer solution.)
#
# The Kussner-Schwarz general solution is used to obtain bound vorticity, 
# pressure distributions, lift and moments from an upwash distribution. It can
# therefore be used to solve operator K_0. 
#
# Use:
# wing = make_rectangular(StraightAnalyticWing, 4, 4) #AR=4, span=4
# prob = GuermondUnsteady2(1, wing) # omega = 1
# compute_collocation_points!(prob)
# compute_fourier_terms!(prob)
# Ys = collect(-1.99:0.01:1.99)
# bvs = map(y->bound_vorticity(prob, y), Ys) # Returns complex!
# cls = map(y->lift_coefficient(prob, y), Ys) # Returns complex!
# plot(Ys, abs.(bvs), label="Abs(Bound vorticity)")
# plot(Ys, abs.(cls), label="Abs(2D lift coeff)")
#
# Copyright HJA Bird 2020
#
#==============================================================================#

mutable struct GuermondUnsteady2
    angular_fq :: Real              # in rad / s
    free_stream_vel :: Real

    wing :: StraightAnalyticWing
    amplitude_fn :: Function        # Amplitude of oscillation wrt/ span pos.
    pitch_plunge :: Int64           # Plunge = 3, Pitch = 5. Otherwise invalid.

    chord_len_scale :: Float64
    span_len_scale :: Float64
    guermond_k :: Float64       # This is uniform over the wing!
    guermond_nu :: Float64

    symmetric :: Bool   # Do we have a symmetric problem?
    sinusoidal_downwash :: Bool # Sears like or uniform induced downwash?
    num_terms :: Int64
    fourier_terms :: Vector{ComplexF64}     # Bound vorticity distribution.
    collocation_points :: Vector{Float64}   # Collocation points in [0, pi]

    function GuermondUnsteady2(
        angular_fq::Real, wing::StraightAnalyticWing;
        free_stream_vel ::Real = 1, amplitude_fn::Function=x->1,
        pitch_plunge::Int=3, chord_len_scale::Real=NaN,
        span_len_scale::Real=NaN, guermond_k::Real=NaN,
        guermond_nu::Real=NaN, num_terms::Int=8,
        fourier_terms=zeros(ComplexF64, 0), collocation_points=zeros(0),
        symmetric=true, sinusoidal_downwash=true)

        chord_len_scale = (chord_len_scale == chord_len_scale ?
            chord_len_scale : wing.chord_fn(0))
        span_len_scale = (span_len_scale == span_len_scale ?
            span_len_scale : wing.semispan)
        guermond_k = (guermond_k == guermond_k ?
            guermond_k : angular_fq * chord_len_scale / (2*free_stream_vel))
        guermond_nu = (guermond_nu == guermond_nu ?
            guermond_nu : angular_fq * span_len_scale / free_stream_vel)

        return new(Float64(angular_fq), Float64(free_stream_vel), wing, 
            amplitude_fn, pitch_plunge, chord_len_scale, span_len_scale,
            guermond_k, guermond_nu, symmetric, sinusoidal_downwash,
            num_terms, fourier_terms, collocation_points)
    end
end

# Mappings from Global to normalised coordinates
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

function j_to_n(a::GuermondUnsteady2, j::Integer) :: Integer
    @assert(j >= 1)
    if a.symmetric
        n = 2 * j - 1
    else
        n = j
    end
    return n
end

# Generate collocation points.
function compute_collocation_points!(
    a :: GuermondUnsteady2) :: Nothing

    nt = a.num_terms
    @assert(nt > 0, "Number of terms in GuermondUnsteady2 must be more than 0")
    pos = Vector{Float64}(undef, a.num_terms)
    hpi = pi / 2
    if a.symmetric
        #for i = 1 : nt
        #    pos[i] = pi / (2 * i)
        #end    nt = a.num_terms
        for i = 1 : nt
            pos[i] = (pi * i - hpi) / (2 * nt)
        end
        a.collocation_points = pos
    else
        for i = 1 : nt
            pos[i] = pi / (2 * i)
        end
        for i = 2 : 2 : nt - 1
            pos[i] = pi - pos[i+1]
        end
    end
    a.collocation_points = pos
    return
end

# 3D bound vorticity from fourier series
function gamma_value(a ::GuermondUnsteady2, y :: Real) :: ComplexF64
    @assert(abs(y) <= 1, "Not -1 <= "*string(y)*" <= 1")
    @assert(length(a.collocation_points) == a.num_terms, "a.num_terms does not"*
        " match the number of computed collocation_points. Did you use "*
        "compute_collocation_points!(problem :: GuermondUnsteady2)?")
    @assert(length(a.fourier_terms) == a.num_terms, "a.num_terms does not"*
        " match the number of computed fourier terms. Did you use "*
        "compute_fourier_terms!(problem :: GuermondUnsteady2)?")
    val = ComplexF64(0)
    theta = acos(y)
    for n = 1 : a.num_terms
        val += a.fourier_terms[n] * sin(j_to_n(a, n) * theta)
    end
    return val
end

# Solve the overall problem using linear algebra.
function compute_fourier_terms!(
    a :: GuermondUnsteady2) :: Nothing

    terms2d = g2d(a)
    terms3d = g3d(a)
    lhs_terms = create_lhs_terms(a)

    sol_mtrx = lhs_terms .+ terms3d ./ aspect_ratio(a.wing) ####################
    g_terms = sol_mtrx \ terms2d
    a.fourier_terms = g_terms
    return
end

# The G values for the 2D problem.
function g2d(a :: GuermondUnsteady2) :: Vector{ComplexF64}
    ys = cos.(a.collocation_points)
    Ys = a.wing.semispan .* ys
    gammas = zeros(ComplexF64, a.num_terms)
    U = a.free_stream_vel
    guerk = a.guermond_k

    for i = 1 : length(ys)
        chord = a.wing.chord_fn(Ys[i])
        k = a.angular_fq * chord / (2* a.free_stream_vel)   # Local k
        amp = a.amplitude_fn(Ys[i])
        if a.pitch_plunge == 3
            uw = make_plunge_function(HarmonicUpwash2D, amp, k;
                free_stream_vel = a.free_stream_vel, semichord=chord/2)
        elseif a.pitch_plunge == 5
            uw = make_pitch_function(HarmonicUpwash2D, amp, k;
                free_stream_vel = a.free_stream_vel, semichord=chord/2)
        end
        bv = bound_vorticity(uw)
        gammas[i] = bv
    end
    return gammas
end

# The matrix representing the sin(j * acos(theta_i)).
function create_lhs_terms(a :: GuermondUnsteady2) :: Matrix{ComplexF64}
    t = zeros(ComplexF64, a.num_terms, a.num_terms)
    for i = 1 : a.num_terms
        for j = 1 : a.num_terms
            t[i, j] = sin(j_to_n(a, j) * a.collocation_points[i])
        end
    end
    return t
end

# Compute integrals of the K_1 operator applied to the sin distribution.
function g3d(a::GuermondUnsteady2) :: Matrix{ComplexF64}
    t = zeros(ComplexF64, a.num_terms, a.num_terms)
    for i = 1 : a.num_terms
        for j = 1 : a.num_terms
            t[i, j] = g3d_term(a, j_to_n(a, j), a.collocation_points[i])
        end
    end
    return t
end

# Actually calculate the g3d integral.
function g3d_term(a::GuermondUnsteady2, 
    n :: Integer, theta :: Real) :: ComplexF64

    y = cos(theta)
    np = 50
    sinnacosy = sin(n * theta)

    # The non-oscillatory term.
    f1 = eta -> sin(n * acos(eta)) / sqrt(1 - eta^2)
    function f1p(eta)
        ti1n = n * cos(n * acos(eta))
        ti1d = eta^2 - 1
        ti2n = eta * sin(n * acos(eta))
        ti2d = (1-eta^2)^(3/2)
        return ti1n / ti1d + ti2n / ti2d
    end
    nonosclterm = integrate_finite_part_chebyshev2(f1, f1p, y; n=np)

    # The inner integral of the oscillatory term
    function inner_integral_fx(eta, v)
        ti1 = sin(n * acos(eta)) / sqrt(1-eta^2)
        ti2 = 1 + v / sqrt(v^2 + (y - eta)^2)
        ti3 = -2 * sinnacosy / (pi * v)
        return ti1 * ti2 + ti3
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
                x->inner_integral_fxp(x, v), y; n=np)
        return ti1
    end
    tosccl = filon_quadrature_exp(inner_terms, a.guermond_nu, 
        -Inf, -1e-2; finite=true)
    total_term = nonosclterm / (4 * pi) - im * a.guermond_nu * tosccl / (4 * pi)
    total_term = total_term * exp(2 * im * a.guermond_k)
    
    chord = a.wing.chord_fn(y * a.wing.semispan)
    omega = a.angular_fq
    U = a.free_stream_vel
    if a.sinusoidal_downwash
        uw = make_sinusoidal_gust_function(HarmonicUpwash2D,
            total_term, 2*a.guermond_k; free_stream_vel= a.free_stream_vel, 
            semichord= chord / 2 )
    else
        uw = make_plunge_function(HarmonicUpwash2D, total_term / a.angular_fq, 
            omega * chord / (2*U);
            free_stream_vel = a.free_stream_vel, semichord=chord/2)
    end
    bv = bound_vorticity(uw)
    return bv
end

#= POST-PROCESSING ----------------------------------------------------------=#
function bound_vorticity(a::GuermondUnsteady2, Y::Real; order::Int=1) :: ComplexF64
    @assert(abs(Y) <= a.wing.semispan)
    @assert(order >= 0, "Positive order required")
    @assert(order < 2, "Only programmed to order 1!")
    # If order = 0, we compute it from 2D
    if order == 0
        uw = upwash_distribution(a, Y; order=0)
        bv = bound_vorticity(uw)
    elseif order == 1
        # Otherwise we compute from the G fourier approximation
        chordte = a.wing.chord_fn(Y) / 2
        bv = gamma_value(a, Y/a.wing.semispan)
    end
    return bv
end

function upwash_distribution(a::GuermondUnsteady2, Y::Real; order::Int=1) :: HarmonicUpwash2D
    #= The upwash distribution is a combination of the 2D upwash and, if
    order = 1, the 3D correction. Input are given with respect to unscaled
    coordinates and the output is also with respect to unscaled coordinates =#

    @assert(abs(Y) <= a.wing.semispan)
    @assert(order >= 0, "Positive order required")
    @assert(order < 2, "Only programmed to order 1!")
    semispan = a.wing.semispan
    U = a.free_stream_vel
    chord = a.wing.chord_fn(Y)
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
    # 1st order: 3D - the more complicated bit. We could work out the
    # strength of the sinusoidal downwash all over again using the big integral
    # expression, but I don't want to program that, so solve instead:
    if order > 0 # or == 1
        bv = gamma_value(a, Y/semispan)
        # 2D
        bv2d = bound_vorticity(uw)
        # 3D
        if a.sinusoidal_downwash
            unit_uw3d = make_sinusoidal_gust_function(HarmonicUpwash2D,
                1, 2*a.guermond_k; free_stream_vel= a.free_stream_vel, 
                semichord= chord / 2 )
        else
            unit_uw3d = make_plunge_function(HarmonicUpwash2D, 1 / a.angular_fq, 
            2*a.guermond_k; free_stream_vel = a.free_stream_vel, semichord=chord/2)
        end
        unit_bv3d = bound_vorticity(uw)
        coeff = (bv - bv2d) / unit_bv3d
        uw = uw + (coeff * unit_uw3d)
    end
    return uw
end

function lift_coefficient(
    a :: GuermondUnsteady2, Y :: Real; order::Int=1) :: ComplexF64

    @assert(order >= 0, "Order must be more than or equal to 0")
    @assert(order < 2, "Only implemented to order 1")
    if abs(Y) >= a.wing.semispan
        cl =  ComplexF64(0)
    else
        uw = upwash_distribution(a, Y; order=order)
        cl = lift_coefficient(uw)
    end
    return cl
end

function lift_coefficient(
    a :: GuermondUnsteady2; order::Int=1) :: ComplexF64

    s = a.wing.semispan
    points, weights = FastGaussQuadrature.gausslegendre(50)
    points, weights = linear_remap(points, weights, -1, 1, -s, s)
    lctc = Y->lift_coefficient(a, Y; order=order) * a.wing.chord_fn(Y)
    CL = sum(weights .* map(lctc, points)) / area(a.wing)
    return CL
end

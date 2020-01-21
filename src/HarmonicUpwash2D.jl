#   
# HarmonicUpwash2D.jl
#
# An harmonically oscillating upwash on an aerofoil represented in terms of 
# fourier coeffs with Kussner-Schwarz solns.
#
# IE: An upwash is defined as v(theta,t) = v_0(theta) exp(i omega t) where
# v_0 = -U (P_0 + 2 sum_1^N P_n cos(n theta)), where the aerofoil is in 
# x = [-l,l] = [ l cos(pi), l cos(0)].
#
# Copyright HJA Bird 2019
#
#==============================================================================#

mutable struct HarmonicUpwash2D
    free_stream_vel :: Real
    omega :: Real
    semichord :: Real
    p0 :: ComplexF64
    pn :: Vector{ComplexF64}

    function HarmonicUpwash2D(free_stream_vel :: Real,
        omega ::Real, semichord :: Real,
        p0 :: Number, pn :: Vector{<:Number})

        @assert(free_stream_vel > 0, "Free stream vel must be postive")
        @assert(omega > 0, "angular frequency must be postive")
        @assert(semichord > 0, "semichord must be postive")
        @assert(isfinite(p0))
        @assert(all(isfinite.(pn)))
        return new(Float64(free_stream_vel),
            omega, semichord,
            ComplexF64(p0), Vector{ComplexF64}(pn))
    end
end

function (a::HarmonicUpwash2D)(theta :: Real)
    @assert(0 <= theta <= pi, "theta should be in (0, pi)")
    ret = -a.free_stream_vel * (a.p0 + 
        2*mapreduce(n->a.pn[n] * cos(n * theta), +, 1:length(a.pn); init=Compex64(0)))
    @assert(isfinite(ret), "non-finite upwash.")
    return ret
end

function (a::HarmonicUpwash2D)(thetas ::Vector{<:Real})
    return map(theta->a(theta), thetas)
end

function make_plunge_function(::Type{HarmonicUpwash2D}, 
    amplitude::Number, k::Real; 
    free_stream_vel::Real=1, semichord::Real=0.5)
    @assert(free_stream_vel > 0)
    @assert(isfinite(amplitude))
    @assert(k > 0, "Reduced frequency must be positive")
    @assert(semichord > 0, "Semichord must be positive")
    omega = free_stream_vel * k / semichord
    ret = HarmonicUpwash2D(
        free_stream_vel, 
        omega,
        semichord, 
        -im * omega * amplitude / free_stream_vel, ComplexF64[])
    # No pn terms are needed.
    return ret
end

function make_pitch_function(::Type{HarmonicUpwash2D}, 
    amplitude::Number, k::Real; 
    free_stream_vel::Real=1, semichord::Real=0.5)
    @assert(free_stream_vel > 0)
    @assert(isfinite(amplitude))
    @assert(k > 0, "Reduced frequency must be positive")
    @assert(semichord > 0, "Semichord must be positive")
    omega = free_stream_vel * k / semichord
    ret = HarmonicUpwash2D(
        free_stream_vel, 
        omega,
        semichord, 
        amplitude, 
        [-amplitude * im * k / (2)])
    # Only 1 pn term needed.
    return ret
end

function make_sinusoidal_gust_function(::Type{HarmonicUpwash2D}, 
    amplitude::Number, k::Real; 
    free_stream_vel::Real=1, semichord::Real=0.5,
    number_of_terms::Integer=20)
    @assert(free_stream_vel > 0)
    @assert(isfinite(amplitude))
    @assert(k > 0, "Reduced frequency must be positive")
    @assert(semichord > 0, "Semichord must be positive")
    @assert(number_of_terms > 0, "Number of terms must be postive integer")
    omega = free_stream_vel * k / semichord
    #= OLD CODE THAT I BELIEVE TO BE WONG.
    p0_term = -amplitude * SpecialFunctions.besselj(0, k) / free_stream_vel
    pn_terms = map(
        n->-2 * (-im)^n * amplitude * SpecialFunctions.besselj(n, k) / free_stream_vel,
        1:number_of_terms-1)        =#
    p0_term = amplitude * SpecialFunctions.besselj(0, k) / free_stream_vel
    pn_terms = map(
        n-> (-im)^n * amplitude * SpecialFunctions.besselj(n, k) / free_stream_vel,
        1:number_of_terms-1)
    
    ret = HarmonicUpwash2D(
        free_stream_vel, 
        omega,
        semichord, 
        p0_term, 
        pn_terms)
    # Only 1 pn term needed.
    return ret
end

function assert_compatible(x::HarmonicUpwash2D, y::HarmonicUpwash2D)
    @assert(x.free_stream_vel == y.free_stream_vel, "Non-matching free_stream_vel")
    @assert(x.omega == y.omega, "Non-matching angular frequencies: x.omega ="*
        string(x.omega)*" =/= "*string(y.omega)*" = y.omega")
    @assert(x.semichord == y.semichord, "Non-matching semichords")
    return 
end

function Base.:+(x::HarmonicUpwash2D, y::HarmonicUpwash2D)
    assert_compatible(x, y)
    pn = zeros(Complex{Float64}, max(length(x.pn), length(y.pn)))
    pn[1:length(x.pn)] += x.pn
    pn[1:length(y.pn)] += y.pn
    return HarmonicUpwash2D(x.free_stream_vel, x.omega, x.semichord,
        x.p0 + y.p0, pn)
end

function Base.:-(x::HarmonicUpwash2D, y::HarmonicUpwash2D)
    assert_compatible(x, y)
    pn = zeros(Complex{Float64}, max(length(x.pn), length(y.pn)))
    pn[1:length(x.pn)] += x.pn
    pn[1:length(y.pn)] -= y.pn
    return HarmonicUpwash2D(x.free_stream_vel, x.omega, x.semichord,
        x.p0 - y.p0, pn)
end

function Base.:*(x::HarmonicUpwash2D, y::Number)
    pn = zeros(ComplexF64, length(x.pn))
    pn = x.pn .* y
    return HarmonicUpwash2D(x.free_stream_vel, x.omega, x.semichord,
        x.p0 * y, pn)
end

function Base.:*(y::Number, x::HarmonicUpwash2D)
    return x*y
end

function Base.:/(x::HarmonicUpwash2D, y::Number)
    return x*(1/y)
end

function pressure_coefficient(a::HarmonicUpwash2D, x::Real)
    @assert(-a.semichord < x <= a.semichord, 
        "x must be in [-semichord, semichord)")
    theta = acos(x/a.semichord)
    k = a.omega * a.semichord / a.free_stream_vel
    pns = a.pn
    pns = length(pns) == 0 ? zeros(1) : pns
    a0 = theodorsen_fn(k) * (a.p0 + pns[1]) - pns[1]
    cp = 4 * a0 * tan(theta / 2)
    if length(pns) > 1
        a1 = im * k /2 * a.p0 + pns[1] - im * k /2 * pns[2]
        cp += 8 * a1 * sin(theta)
    end
    for i = 2 : length(pns)-1
        an =  im * k /(2 * i) * pns[i-1] + pns[i] - im * k /(2 *i) * pns[i+1]
        cp += 8 * an * sin(i * theta)
    end
    return cp
end

function lift_coefficient(a::HarmonicUpwash2D)
    p0 = a.p0
    p1 = length(a.pn) > 0 ? a.pn[1] : 0
    p2 = length(a.pn) > 1 ? a.pn[2] : 0
    k = a.omega * a.semichord / a.free_stream_vel
    ck = theodorsen_fn(k)
    cl = 2 * pi * ((p0 + p1) * ck + (p0 - p2) * im * k / 2)
    return cl
end

function moment_coefficient(a::HarmonicUpwash2D)
    # About the midchord!
    p0 = a.p0
    p1 = length(a.pn) > 0 ? a.pn[1] : 0
    p2 = length(a.pn) > 1 ? a.pn[2] : 0
    p3 = length(a.pn) > 2 ? a.pn[3] : 0
    k = a.omega * a.semichord / a.free_stream_vel
    l = a.semichord
    ck = theodorsen_fn(k)
    cm =  pi / 2 * (p0 * ck - p1 * (1 - ck) - (p1 - p3) * im * k / 4 - p2)
    return cm
end

function bound_vorticity(a::HarmonicUpwash2D)
    U = a.free_stream_vel
    k = a.omega * a.semichord / U
    h21 = SpecialFunctions.hankelh2(1, k)
    h20 = SpecialFunctions.hankelh2(0, k)
    p0 = a.p0
    p1 = length(a.pn) > 0 ? a.pn[1] : ComplexF64(0)
    t1 = 4 * U^2 * im * exp(-im * k) / a.omega
    t2n = p0 - p1
    t2d = im * h20 + h21
    ret = t1 * t2n / t2d
    return ret
end

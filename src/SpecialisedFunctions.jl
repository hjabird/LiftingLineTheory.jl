#
# SpecialisedFunctions.jl
#
# Copyright HJA Bird 2019
#
#============================================================================#

import SpecialFunctions
import FastGaussQuadrature
import LinearAlgebra
import PyPlot

#= Aerodynamics functions --------------------------------------------------=#
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

"""
Generate a Wagner function of using a variable number of terms.

A sum of a_i * exp(b_i * s) terms can be used to approximate Wagner's function.
This is done by taking matching the fourier transform of the above with
Theodorsen's function.

The method currently used is very sensitive to its arguments and sometimes 
produces complete garbage.
"""
function create_wagner_fn(num_terms :: Int, k_max :: Real)
    @assert(num_terms > 0)
    I = num_terms
    lim_k_to_inf = 0.5  # Limit as k->infinity for theodorsen_fn
    lim_k_to_zero = 1   # for theodorsen fn.

    ai = Vector{Float64}(undef, I)
    bi = Vector{Float64}(undef, I)
    bi[1] = 0       # Required to satisfy k->0 limit
    ai[1] = lim_k_to_zero
    # Arbitrarily set
    bi[2:end] = - collect(1:I-1) * k_max / (I - 1)
    # Frequency collocation points - spread over the curve created on complex 
    # plane
    kn = collect( i / (2 * sqrt(I-2)) for i = 1 : I-2)
    mat_inp = [(b, k) for k in kn, b in bi[2:end]]
    matrix = Matrix{Float64}(undef, I-1, I-1)
    matrix[1:I-2,:] = map(
        x -> x[2]^2 / (x[1]^2 + x[2]^2),
        mat_inp)
    matrix[end,:] .= 1
    rhs_vec = vcat(real.(theodorsen_fn.(kn)), lim_k_to_inf) .- lim_k_to_zero
    ai[2:end] = matrix \ rhs_vec
    function wagner_fn(s :: Real)
        if s >= 0
            return mapreduce(
                x -> x[1] * exp(x[2] * s),
                +,
                zip(ai, bi) )
        else
            return 0
        end
    end
    println("Ai = \t", ai)
    println("Bi = \t", bi)
    return wagner_fn
end

function create_wagner_fn2(num_terms :: Int, k_max :: Real)
    @assert(num_terms > 0)
    I = num_terms
    lim_k_to_inf = 0.5  # Limit as k->infinity for theodorsen_fn
    lim_k_to_zero = 1   # for theodorsen fn.

    ai = Vector{Float64}(undef, I)
    bi = Vector{Float64}(undef, I)
    bi[1] = 0       # Required to satisfy k->0 limit
    ai[1] = lim_k_to_zero
    # Arbitrarily set
    bi[2:end] = - collect(1:I-1) * k_max / (I - 1)
    # Frequency collocation points - spread over the curve created on complex 
    # plane
    kn = collect( i / (8 * sqrt(I-2)) for i = 1 : I-2)
    mat_inp_r = [(b, k) for k in kn[1:2:end], b in bi[2:end]]
    mat_inp_i = [(b, k) for k in kn[2:2:end], b in bi[2:end]]
    # Mat goes [real, imag, real, imag, .., lim_k_inf(real)]
    println("kn = \t\t", kn)
    matrix = Matrix{Float64}(undef, I-1, I-1)
    matrix[1:2:I-2,:] = map(                # Real Part
        x -> x[2]^2 / (x[1]^2 + x[2]^2),
        mat_inp_r)
    matrix[2:2:I-2,:] = map(                # Imaginary part
        x -> -x[2]*x[1] / (x[1]^2 + x[2]^2),
        mat_inp_i)
    matrix[end,:] .= 1
    rhs_vec = Vector{Float64}(undef, I-1)
    a1_vec = Vector{Float64}(undef, I-1)
    a1_vec[1:2:end] .= lim_k_to_zero    # real
    a1_vec[end] = lim_k_to_zero         # lim_k_zero
    a1_vec[2:2:end-1] .= 0              # imag
    rhs_vec[1:2:end-1] = real.(theodorsen_fn.(kn[1:2:end])) # real
    rhs_vec[2:2:end-1] = imag.(theodorsen_fn.(kn[2:2:end])) # imag
    rhs_vec[end] = lim_k_to_inf         # lim_k_inf

    println("A1 = \t\t", a1_vec)
    println("rsh= \t\t", rhs_vec)
    println("mat= \t\t", matrix)
    println("cond(mat) = ", LinearAlgebra.cond(matrix))

    ai[2:end] = matrix \ (rhs_vec - a1_vec)
    function wagner_fn(s :: Real)
        if s >= 0
            return mapreduce(
                x -> x[1] * exp(x[2] * s),
                +,
                zip(ai, bi) )
        else
            return 0
        end
    end
    println("Ai = \t", ai)
    println("Bi = \t", bi)

    ks = collect(0.001: 0.002 : 4)
    tr = real.(theodorsen_fn.(ks))
    ti = imag.(theodorsen_fn.(ks))
    fr = real.(map(k->mapreduce(x->im*k*x[1]/(im*k-x[2]), +, zip(ai, bi)), ks))
    fi = imag.(map(k->mapreduce(x->im*k*x[1]/(im*k-x[2]), +, zip(ai, bi)), ks))
    trd = real.(theodorsen_fn.(kn))
    tid = imag.(theodorsen_fn.(kn))
    frd = real.(map(k->mapreduce(x->im*k*x[1]/(im*k-x[2]), +, zip(ai, bi)), kn))
    fid = imag.(map(k->mapreduce(x->im*k*x[1]/(im*k-x[2]), +, zip(ai, bi)), kn))
    PyPlot.figure()
    PyPlot.plot(tr, ti, "k-")
    PyPlot.plot(fr, fi, "r-")
    println(trd)
    PyPlot.plot(trd, tid, "kx")
    PyPlot.plot(frd, fid, "rx")

    return wagner_fn
end

"""
R.T. Jones' approximation of Wagner's function.

Argument of s is normalised. For example s = U * t / b where 
U is free stream vel, t is time since the step change and b is the semichord
of the wing section.
"""
function wagner_fn(s :: Real)
    return 1 - 0.165 * exp(-0.0455*s) - 0.335 * exp(-0.3*s)
end

function approximate_interaction_wrt_srf(
    a :: HarmonicULLT)

    ac = deepcopy(a)    # Avoid messing with original
    srfs = vcat(collect(0.01: 0.05 : 1), collect(1.25: 0.25 : 4), collect(5:1.:8))
    compute_collocation_points!(a)
    swps = map(x->theta_to_y(a, x), a.collocation_points)
    # Step 1 collect data.
    zs = Matrix{Complex{Float64}}(undef, length(srfs), length(swps))
    for i = 1 : length(srfs)
        fq = srfs[i] * a.free_stream_vel / a.wing.semispan
        a.angular_fq = fq
        compute_collocation_points!(a)
        compute_fourier_terms!(a)
        zs[i, :] = map(y->f_eq(a, y), swps)
    end

    PyPlot.figure()
    colours = "rbgymckrbgymck"
    markers ="xo^.v+"
    for i = 1 : length(swps)
        spline_r = Dierckx.Spline1D(srfs, real.(zs[:, i]); bc="extrapolate")
        spline_i = Dierckx.Spline1D(srfs, imag.(zs[:, i]); bc="extrapolate")
        kr = vcat(collect(0:0.02:4), collect(4.2:0.2:8))#, collect(10 : 1. :20))#, collect(50 :50. :250))
        kc = [0.8, 4]
        PyPlot.plot(spline_r.(kr), spline_i.(kr), colours[(i-1)%14 + 1] * "-d")
        for j = 1 : length(kc)
            PyPlot.plot(spline_r(kc[j]), spline_i(kc[j]), colours[(i-1)%14 + 1] * markers[(j-1)%6 + 1])
            PyPlot.plot(spline_r(kc[j]), spline_i(kc[j]), colours[(i-1)%14 + 1] * markers[(j-1)%6 + 1])
        end
    end
    return
end

#= Mappings from Real->Real ------------------------------------------------=#
function linear_remap(
    pointin :: Number,   weightin :: Number,
    old_a :: Number,     old_b :: Number,
    new_a :: Number,     new_b :: Number )

    dorig = (pointin - old_a) / (old_b - old_a)
    p_new = new_a + dorig * (new_b - new_a)
    w_new = ((new_b - new_a) / (old_b - old_a)) * weightin
    return p_new, w_new
end

function telles_cubic_remap(
	pointin :: Number, 	weightin :: Number,
	pointsingular :: Number,
	lima:: Number, 		limb :: Number)
	# Remap problem to [-1, 1]
	p, w = linear_remap(pointin, weightin, lima, limb, -1, 1)
	ps, ~ = linear_remap(pointsingular, 1, lima, limb, -1, 1)
	# Telles cubic...
	# Remapped singularity position
	sprm = cbrt((ps - 1)*(ps + 1)*(ps + 1)) +
		cbrt((ps -1)*(ps - 1)* (ps + 1)) + ps
	# Allows us to calculate new point and weight positions
	pn = ((p - sprm)^3 + sprm * (sprm^2 + 3)) / (3 * sprm^2 + 1)
	wn = w * 3 * (sprm - p)^2 / (3 * sprm^2 + 1)	
	# Map solution back to original domain
	p, w = linear_remap(pn, wn, -1, 1, lima, limb)
	return p, w	
end

#= Laplace transform -------------------------------------------------------=#
function laplace(
    function_in_t :: Function, s :: Real)

    nodes, weights = FastGaussQuadrature.gausslaguerre(100)
    integrand = x -> function_in_t(x / s) / s
    evals = map(integrand, nodes)
    integral = sum(weights .* evals)
    return integral
end

#= Inverse Laplace transforms ----------------------------------------------=#
function gaver_stehfest(
    function_in_s :: Function, t :: Real, N :: Int)

    # Gaver-Stehfest method.
    @assert(hasmethod(function_in_s, (Float64,)), "Input method will not "*
        "accept a F64 as argument in for Gaver-Stehfest inverse laplace.")
    @assert(N > 0, "Taylor approximation of transform must have positive"*
        " number of terms. (Given N <= 0)")
    @assert((N%2) == 0, "N (number of terms) must be even.")

    function Vk(k :: Int, N :: Int)
        N2 = N / 2
        term1i = (-1)^(k + N2)
        term2i = mapreduce(
            x->x^(N2) * factorial(2 * x) /
                (factorial(N2 - x) * factorial(x) * factorial(x - 1) *
                factorial(k - x) * factorial(2*x - k)),
            +,
            collect(floor((k+1)/2) : minimum((k, N/2)))
        )
        return term1i * term2i
    end
    term1 = log(2) / t
    term2 = mapreduce(
        x-> Vk(x, N) * function_in_s(x * term1),
        +,
        collect(1 : N)
    )
    return term1 * term2
end

function glauert_integral(k :: Real, alpha :: Real)
    return pi * sin(abs(k)* alpha) / sin(alpha)
end

#= Special functions -------------------------------------------------------=#

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

# END SpecialisedFunctions.jl
#============================================================================#

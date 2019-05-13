#
# FDTDTExpApprox.jl
#
# Exponential approximant (ie sum H(t) * t * a_i exp(b_i * t)), and an
# interpolator for the approximant.
#
# Copyright HJA Bird 2019
#
#==============================================================================#

import Base

mutable struct FDTDTExpApprox{T}
    a_i :: Vector{T}
    b_i :: Vector{T}
end

function td_eval(a::FDTDTExpApprox{T}, t::Real) where T<:Real
    @assert(all(a.b_i .< 0), "All be values must be less than 0")
    @assert(length(a.b_i) == length(a.a_i))
    return mapreduce(x->x[1] * t * exp(x[2] * t), +, zip(a.a_i, a.b_i))
end

function fd_eval(a::FDTDTExpApprox{T}, fq::Real) where T<:Real
    @assert(all(a.b_i .< 0), "All be values must be less than 0")
    @assert(length(a.b_i) == length(a.a_i))
    @assert(length(a.b_i) == length(a.a_i))
    res = mapreduce(
        x->im * fq * x[1] / (im * fq - x[2])^2,
        +, zip(a.a_i, a.b_i); init=T(0))
    return res
end

function td_derivative(a::FDTDTExpApprox{T}) where T<:Real
    @assert(all(a.b_i .< 0), "All be values must be less than 0")
    @assert(length(a.b_i) == length(a.a_i))
    c = map(x->(x[1]*x[2], x[1]), zip(a.a_i, a.b_i))
    d = map(x->(x,x), a.b_i)
    res = map(x->
            (FDTDTExpApprox(x[1][1], x[2][1]),
            FDTDExpApprox(x[1][2], x[2][2])),
            zip(c, d))
    return res
end

function td_integral(a::FDTDTExpApprox{T}) where T<:Real
    @assert(all(a.b_i .< 0), "All be values must be less than 0")
    @assert(length(a.b_i) == length(a.a_i))
    c = map(x->(x[1]/x[2], -x[1]/x[2]^2), zip(a.a_i, a.b_i))
    d = map(x->(x,x), a.b_i)
    res = map(x->
            (FDTDTExpApprox(x[1][1], x[2][1]),
            FDTDExpApprox(x[1][2], x[2][2])),
            zip(c, d))
    return res
end

function duhamel_int(step_res::FDTDExpApprox{T}, inducer::FDTDTExpApprox{R}, t::Real) where {T<:Real, R<:Real}
    a = step_res
    b = inducer
    function integral(i, j)
        t1 = a.a_i[i] * b.a_i[j] * exp(a.b_i[i] * t)
        if a.b_i[i] != b.b_i[j]
            den = b.b_i[j] - a.b_i[i]
            t211 = 1 / den - b.b_i[j] / den^2
            t212 = exp(den * t) - 1
            t21 = t211 * t212
            t22 = b.b_i[j] * t * exp(den * t) / den
            ret = t1 * (t21 + t22)
        else
            ret = t1 * (t + b.b_i[j] * t^2 / 2)
        end
        return ret
    end

    result = mapreduce(
        i->mapreduce(
            j->integral(i, j),
            +, 1 : num_terms(b); init=0
        ), +, 1 : num_terms(a); init=0
    )
    return result;
end

function Base.:*(a::FDTDTExpApprox{T}, b::FDTDTExpApprox{R}) where {T<:Real, R<:Real}
    @error("No can do. Need to define a * t^2 * exp(b * t) first.")
    return
end

function num_terms(a::FDTDTExpApprox{T}) where T<:Real
    @assert(length(a.a_i) == length(a.b_i), "Coefficient vectors are "*
        "different lengths.")
    return length(a.a_i)
end

#=-- Interpolation of FDTDExpApprox ---------------------------------=#
mutable struct FDTDTExpApproxInterp
    positions :: Vector{Float64}
    approximants :: Vector{FDTDTExpApprox{Float64}}
    semispan :: Real
    valid :: Bool
    a_i_interp :: Vector{CubicSpline{Float64}}
    b_i_interp :: Vector{CubicSpline{Float64}}
end

function FDTDTExpApproxInterp(
    positions :: Vector{Float64},
    approximants :: Vector{FDTDTExpApprox{Float64}},
    semispan :: Real) where {T<:Real}

    @assert(length(positions) == length(approximants),
        "Length of position and approximant vectors must be the same."*
        " Positions has length "*string(length(positions))*" and "*
        " approximants has length "*string(length(approximants))*"." )
    @assert(semispan > 0,
        "Semispan of the wing must be positive.")
    @assert(all(map(x->abs(x) < semispan, positions)),
        "All positions must be within the semispan of the wing.")
    @assert(all(map(num_terms, approximants) .== num_terms(approximants[1])),
        "All approximant must have equal numbers of terms. Num terms are "*
        string(map(num_terms, approximants))*".")
    @assert(issorted(positions),
        "Position[i] < Position[i+1] is required.")
        FDTDTExpApproxInterp(positions, approximants, semispan, false,
        Vector{CubicSpline{Float64}}(undef, 0),
        Vector{CubicSpline{Float64}}(undef, 0))
end

function compute_interpolation!(a::FDTDTExpApproxInterp)
    @assert(length(a.positions) > 1, "length(positions) <=1: need something to"*
        " interpolate.")
    @assert(length(a.positions) == length(a.approximants),
        "the length of the position and approximants vectors must match.")
    @assert(all(map(num_terms, a.approximants) .== num_terms(a.approximants[1])),
        "All approximant must have equal numbers of terms. Num terms are "*
        string(map(num_terms, a.approximants))*".")

    n_terms = num_terms(a.approximants[1])
    num_positions = length(a.positions)
    a_i = Vector{CubicSpline{Float64}}(undef, n_terms)
    b_i = Vector{CubicSpline{Float64}}(undef, n_terms)
    for i = 1 : n_terms
        an = map(x->x.a_i[i], a.approximants)
        bn = map(x->x.b_i[i], a.approximants)
        a_i[i] = CubicSpline{Float64}(a.positions, an)
        b_i[i] = CubicSpline{Float64}(a.positions, bn)
    end
    a.a_i_interp = a_i
    a.b_i_interp = b_i
    a.valid = true
    return
end

function interpolate(a::FDTDTExpApproxInterp, x::Real)
    @assert(abs(x) <= a.semispan, "x is outside of semispan")
    if !a.valid
        compute_interpolation!(a)
    end
    a_i = Vector{Float64}(undef, num_terms(a.approximants[1]))
    b_i = Vector{Float64}(undef, num_terms(a.approximants[1]))
    for i = 1 : length(a.a_i_interp)
        a_i[i] = a.a_i_interp[i](x)
        b_i[i] = a.b_i_interp[i](x)
    end
    return FDTDTExpApprox(a_i, b_i)
end

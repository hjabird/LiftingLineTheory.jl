#
# FDTDExpApprox.jl
#
# Exponential approximant (ie sum H(t) * a_i exp(b_i * t)), and an interpolator 
# for the approximant.
#
# Copyright HJA Bird 2019
#
#==============================================================================#

import Base

mutable struct FDTDExpApprox{T}
    a_i :: Vector{T}
    b_i :: Vector{T}
end

function td_eval(a::FDTDExpApprox{T}, t::Real) where T<:Real
    @assert(length(a.b_i) == length(a.a_i))
    return mapreduce(x->x[1] * exp(x[2] * t), +, zip(a.a_i, a.b_i))
end

function fd_eval(a::FDTDExpApprox{T}, fq::Real) where T<:Real
    @assert(all(map(x->x <= 0, a.b_i)))
    @assert(fq > 0)
    @assert(length(a.b_i) == length(a.a_i))
    return mapreduce(x->im * fq * x[1] / (im * fq - x[2]), +, zip(a.a_i, a.b_i))
end

function td_derivative(a::FDTDExpApprox{T}) where T<:Real
    c = map(x->x[1]*x[2], zip(a.a_i, a.b_i))
    return FDTDExpApprox{T}(c, a.b_i)
end

function td_integral(a::FDTDExpApprox{T}) where T<:Real
    c = map(
        x->x[2] == 0 ? x[1] : x[1] / x[2], zip(a.a_i, a.b_i) )
    return FDTDExpApprox(c, a.b_i)
end

function duhamel_int(step_res::FDTDExpApprox{T}, inducer::FDTDExpApprox{R}, t::Real) where {T<:Real, R<:Real}
    a = step_res
    b = td_derivative(inducer)
    result = mapreduce(
        i->mapreduce(
            j-> a.b_i[i] == b.b_i[j] ? 
                a.a_i[i] * b.a_i[j] * exp(a.b_i[i]) :
                a.a_i[i] * b.a_i[j] * (exp(b.b_i[j] * t) - exp(a.b_i[i] * t)) /
                    (b.b_i[j] - a.b_i[i]),
            +, 1 : num_terms(b)
        ), +, 1 : num_terms(a)
    )
    return result;
end

function Base.:*(a::FDTDExpApprox{T}, b::FDTDExpApprox{R}) where {T<:Real, R<:Real}
    c = mapreduce(
        x->map(i->a.a_i[i] * x, 1:length(a.a_i)),
        vcat,
        b.a_i
    )
    d = mapreduce(
        x->map(i->a.a_i[i] + x, 1:length(a.b_i)),
        vcat,
        b.b_i
    )
    return FDTDExpApprox(c, d)
end

function Base.:+(a::FDTDExpApprox{T}, b::FDTDExpApprox{R}) where {T<:Real, R<:Real}
    na = Vector{typeof(a.a_i+b.a_i)}(undef, 0)
    nb = Vector{typeof(a.b_i+b.b_i)}(undef, 0)
    for i = 1 : num_terms(a)
        push!(na, a.a_i[i])
        push!(nb, a.b_i[i])
    end
    for i = 1 : num_terms(b)
        push!(na, b.a_i[i])
        push!(nb, b.b_i[i])
    end
    return FDTDExpApprox(na, nb)
end

function num_terms(a::FDTDExpApprox{T}) where T<:Real
    @assert(length(a.a_i) == length(a.b_i), "Coefficient vectors are "*
        "different lengths.")
    return length(a.a_i)
end

#=-- Interpolation of FDTDExpApprox ---------------------------------=#
mutable struct FDTDExpApproxInterp
    positions :: Vector{Float64}
    approximants :: Vector{FDTDExpApprox{Float64}}
    semispan :: Real
    valid :: Bool
    a_i_interp :: Vector{LinearInterpolator{Float64}}
    b_i_interp :: Vector{LinearInterpolator{Float64}}
end

function FDTDExpApproxInterp(
    positions :: Vector{Float64},
    approximants :: Vector{FDTDExpApprox{Float64}},
    semispan :: Real)

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
    FDTDExpApproxInterp(positions, approximants, semispan, false,
        Vector{LinearInterpolator{Float64}}(undef, 0), 
        Vector{LinearInterpolator{Float64}}(undef, 0))
end

function compute_interpolation!(a::FDTDExpApproxInterp)
    @assert(length(a.positions) > 1, "length(positions) <=1: need something to"*
        " interpolate.")
    @assert(length(a.positions) == length(a.approximants),
        "the length of the position and approximants vectors must match.")
    @assert(all(map(num_terms, a.approximants) .== num_terms(a.approximants[1])),
        "All approximant must have equal numbers of terms. Num terms are "*
        string(map(num_terms, a.approximants))*".")
    
    n_terms = num_terms(a.approximants[1])
    num_positions = length(a.positions)
    a_i = Vector{LinearInterpolator{Float64}}(undef, n_terms)
    b_i = Vector{LinearInterpolator{Float64}}(undef, n_terms)
    for i = 1 : n_terms
        an = map(x->x.a_i[i], a.approximants)
        bn = map(x->x.b_i[i], a.approximants)
        a_i[i] = LinearInterpolator{Float64}(a.positions, an)
        b_i[i] = LinearInterpolator{Float64}(a.positions, bn)
    end
    a.a_i_interp = a_i
    a.b_i_interp = b_i
    a.valid = true
    return
end

function interpolate(a::FDTDExpApproxInterp, x::Real)
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
    return FDTDExpApprox{Float64}(a_i, b_i)
end

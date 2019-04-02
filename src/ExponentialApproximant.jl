#
# ExponentialApproximant.jl
#
# Copyright HJA Bird 2019
#
#==============================================================================#

import Base

mutable struct ExponentialApproximant{T}
    a_i :: Vector{T}
    b_i :: Vector{T}
end

function td_eval(a::ExponentialApproximant{T}, t::Real) where T<:Real
    @assert(length(a.b_i) == length(a.a_i))
    return mapreduce(x->x[1] * exp(x[2] * t), +, zip(a.a_i, a.b_i))
end

function fd_eval(a::ExponentialApproximant{T}, fq::Real) where T<:Real
    @assert(all(map(x->x <= 0, a.b_i)))
    @assert(fq > 0)
    @assert(length(a.b_i) == length(a.a_i))
    return mapreduce(x->im * fq * x[1] / (im * fq - x[2]), +, zip(a.a_i, a.b_i))
end

function derivative(a::ExponentialApproximant{T}) where T<:Real
    c = map(x->x[1]*x[2], zip(a.a_i, a.b_i))
    return ExponentialApproximant{T}(c, a.b_i)
end

function integral(a::ExponentialApproximant{T}) where T<:Real
    c = map(
        x->x[2] == 0 ? x[1] : x[1] / x[2], zip(a.a_i, a.b_i) )
    return ExponentialApproximant(c, a.b_i)
end

function Base.:*(a::ExponentialApproximant{T}, b::ExponentialApproximant{R}) where {T<:Real, R<:Real}
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
    return ExponentialApproximant(c, d)
end

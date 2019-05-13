#
# FDTDMixedApprox.jl
#
# A linear sum of FD<->TD approximations
#
# Copyright HJA Bird 2019
#
#==============================================================================#

import Base

mutable struct FDTDMixedApprox{T, R}
    exp_approx :: FDTDExpApprox{T}
    texp_approx :: FDTDTExpApprox{R}
end

function td_eval(a::FDTDMixedApprox{T, R}, t::Real) where {T<:Real, R<:Real}
    return td_eval(a.exp_approx, t) + td_eval(a.texp_approx, t)
end

function fd_eval(a::FDTDMixedApprox{T}, fq::Real) where {T<:Real, R<:Real}
    return fd_eval(a.exp_approx, fq) + fd_eval(a.texp_approx, fq)
end

function td_derivative(a::FDTDMixedApprox{T, R}) where {T<:Real, R<:Real}
    texp_deriv = td_derivative(a.texp_appox)
    return FDTDMixedApprox(td_derivative(a.exp_approx) + first(texp_deriv),
        last(texp_deriv))
end

function td_integral(a::FDTDMixedApprox{T, R}) where {T<:Real, R<:Real}
    texp_int = td_integral(a.texp_appox)
    return FDTDMixedApprox(td_integral(a.exp_approx) + first(texp_int),
        last(texp_int))
end

function duhamel_int(step_res::FDTDExpApprox{T}, inducer::FDTDMixedApprox{R, S}, t::Real) where {T<:Real, R<:Real, S<:Real}
    return duhamel_int(step_res, inducer.exp_approx, t) +
        duhamel_int(step_res, inducer.texp_approx, t)
end

function num_terms(a::FDTDMixedApprox{T, R}) where {T<:Real, R<:Real}
    return num_terms(a.exp_approx) + num_terms(a.texp_approx)
end

#=-- Interpolation of FDTDExpApprox ---------------------------------=#
mutable struct FDTDMixedApproxInterp
    exp_interp :: FDTDExpApproxInterp
    texp_interp :: FDTDTExpApproxInterp
end

function FDTDMixedApproxInterp(
    positions :: Vector{Float64},
    approximants :: Vector{FDTDMixedApprox},
    semispan :: Real)

    exp_a = FDTDExpApproxInterp(positions,
        map(x->x.exp_approx, approximants), semispan)
    texp_a = FDTDTExpApproxInterp(positions,
        map(x->x.texp_approx, approximants), semispan)
        FDTDMixedApproxInterp(exp_a, texp_a)
end

function compute_interpolation!(a::FDTDMixedApproxInterp)
    @assert(a.exp_interp.semispan == a.texp_interp.semispan)
    compute_interpolation!(a.exp_interp)
    compute_interpolation!(a.texp_interp)
    return
end

function interpolate(a::FDTDMixedApproxInterp, x::Real)
    @assert(a.exp_interp.semispan == a.texp_interp.semispan)
    exp_a = interpolate(a.exp_interp, x)
    texp_a = interpolate(a.texp_interp, x)
    return FDTDMixedApprox(exp_a, texp_a)
end

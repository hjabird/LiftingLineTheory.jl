#
# CubicSpline.jl
#
# Copyright HJA Bird 2019
#
#============================================================================#

abstract type Interpolator1D end

struct CubicSpline{T} <: Interpolator1D
    xs :: Vector{T}
    ys :: Vector{T}
    second_derivs :: Vector{T}
    deriv_x0 :: Union{T, Nothing}
    deriv_xn :: Union{T, Nothing}
    nat_bc_x0 :: Bool
    nat_bc_xn :: Bool
    
    function CubicSpline{T}(
        xs :: Vector{T}, ys :: Vector{T};
        deriv_x0 :: Union{T, Nothing}=nothing,
        deriv_xn :: Union{T, Nothing}=nothing) where {T <: Real}

        @assert(length(xs) == length(ys),
            "Position (arg1) and value (arg2) vectors must have the "*
            "same length. Lengths are " * string(length(xs)) * " and "*
            string(length(ys))*" respectively.")
        @assert(issorted(xs), "Position vector (arg1) must be sorted "*
            "increasing.")
        
        natbcx0 = deriv_x0 == nothing
        natbcxn = deriv_xn == nothing

        n = length(ys)
        yd2 = Vector{T}(undef, n)
        u = Vector{T}(undef, n)
        if natbcx0
            yd2[1] = u[1] = 0.0 .* ys[1]
        else
            yd2[1] = -0.5;
            u[1] = (3/(xs[2]-xs[1])) * ((ys[2]-ys[1])/(xs[2]-xs[1]) - deriv_x0)
        end
        for i = 1 : n - 2
            sig = (xs[i+1] - xs[i]) / (xs[i+2] - xs[i])
            p = sig * yd2[i] + 2
            yd2[i+1] = (sig - 1)/p
            u[i+1] = (ys[i + 2] - ys[i+1]) / (xs[i+2] - xs[i+1]) -
                (ys[i+1] - ys[i]) / (xs[i+1] - xs[i])
            u[i+1] = (6 * u[i+1] / (xs[i + 2] - xs[i]) - sig * u[i]) / p
        end
        if natbcxn
            qn = un = 0.0
        else
            qn = 0.5
            un = (3.0 / (xs[n] - xs[n-1])) * (deriv_xn - (ys[n] - ys[n-1]) /
                (xs[n] - xs[n-1]))
        end
        yd2[end] = (un - qn * u[n-1]) / (qn * yd2[n-1] + 1)
        for i = n-1 : -1 : 1
            yd2[i] = yd2[i] * yd2[i+1] + u[i]
        end
        new(xs, ys, yd2, deriv_x0, deriv_xn, natbcx0, natbcxn)
    end

end

function (a::CubicSpline)(x :: Real)
    klo = 1
    khi = length(a.xs)
    while khi - klo > 1
        k = Int64(floor((khi + klo) / 2))
        if a.xs[k] > x 
            khi = k
        else
            klo = k
        end
    end
    h = a.xs[khi] - a.xs[klo]
    oa = (a.xs[khi] - x) / h
    ob = (x - a.xs[klo]) / h 
    y = oa * a.ys[klo] + ob * a.ys[khi] +
        (oa * (oa * oa * oa  - 1) * a.second_derivs[klo] +
        ob * (ob * ob * ob  - 1) * a.second_derivs[khi]) *
            (h * h) / 6
    return y
end

struct LinearInterpolator{T} <: Interpolator1D
    xs :: Vector{T}
    ys :: Vector{T}

    function LinearInterpolator{T}(
        xs :: Vector{T}, ys :: Vector{T}) where T <: Real

        @assert(length(xs) == length(ys),
            "Position (arg1) and value (arg2) vectors must have the "*
            "same length. Lengths are " * string(length(xs)) * " and "*
            string(length(ys))*" respectively.")
        @assert(issorted(xs), "Position vector (arg1) must be sorted "*
            "increasing.")

        new(xs, ys)
    end
end

function (a::LinearInterpolator)(x :: Real)
    klo = 1
    khi = length(a.xs)
    while khi - klo > 1
        k = Int64(floor((khi + klo) / 2))
        if a.xs[k] > x 
            khi = k
        else
            klo = k
        end
    end
    h = a.xs[khi] - a.xs[klo]
    oa = (a.xs[khi] - x) / h
    ob = (x - a.xs[klo]) / h 
    y = oa * a.ys[klo] + ob * a.ys[khi]
    return y
end
#
# CurvedAnalyticWing.jl
#
# Not currently in use by anything.
#
# Copyright HJA Bird 2019
#==============================================================================#

mutable struct CurvedAnalyticWing
    straight_geometry :: StraightAnalyticWing
    curvature :: Function
    dcurvature_dy :: Function

    function CurvedAnalyticWing(
        straight_geometry :: StraightAnalyticWing,
        curvature_fn :: Function,
        dcurvature_fn_dy :: Function)
        @assert(hasmethod(curvature_fn, (Float64)), "Must be able to "*
            "evaluate curvature function with respect to single argument.")
        @assert(hasmethod(dcurvature_fn_dy, (Float64)), "Must be able to "*
            "evaluate derivative of curvature function with respect to "*
            "single argument.")
        new(straight_geometry, curvature_fn, dcurvature_fn_dy)
    end
    
    function CurvedAnalyticWing(
        straight_geometry :: StraightAnalyticWing,
        curvature_fn :: Function)
        @assert(hasmethod(curvature_fn, (Float64)), "Must be able to "*
            "evaluate curvature function with respect to single argument.")
        dcurvature_dy = x->0
        try
            dcurvature_dy = x->ForwardDiff.derivative(curvature_fn, x)
            dcurvature_dy(0.0)
        catch
            semispan = straight_geometry.semispan
            epsilon = 1e-7
            dcurvature_dy = x->(
                curvature_fn(x+semispan*epsilon)-curvature_fn(x-semispan*epsilon))/
                (semispan*epsilon)
        end
        new(straight_geometry, curvature_fn, dcurvature_dy)
    end

    function CurvedAnalyticWing(
        straight_geometry :: StraightAnalyticWing)

        curvature_fn = y->0
        @assert(hasmethod(curvature_fn, (Float64)), "Must be able to "*
            "evaluate curvature function with respect to single argument.")
        new(straight_geometry, curvature_fn)
    end
end

#= I'm sure properties are evil, but if you can't beat them, join them... ====#
function Base.getproperty(a::CurvedAnalyticWing, name::Symbol)
    if name in fieldnames(typeof(a))
        getfield(a, name)
    elseif name in fieldnames(typeof(a.straight_geometry))
        getfield(a.source_object, name)
    else
        error("type CurvedAnalyticWing has no field "*string(name))
    end
end

function Base.setproperty!(a::CurvedAnalyticWing, name::Symbol, value)
    if name in fieldnames(typeof(a))
        setfield!(a, name)
    elseif name in fieldnames(typeof(a.straight_geometry))
        setfield!(a.straight_geometry, name, value)
    else
        error("type CurvedAnalyticWing has no field "*string(name))
    end
end

function make_rectangular(
    ::Type{CurvedAnalyticWing}, 
    aspect_ratio :: Real, span :: Real ) 

    fn = y -> span / aspect_ratio
    semispan = span / 2
    return CurvedAnalyticWing(semispan, fn)
end

function make_elliptic(
    ::Type{CurvedAnalyticWing}, 
    aspect_ratio :: Real, span :: Real )

    semispan = span / 2
    # We multiply by two since we want the sum of the LE + TE offsets.
    fn = y -> (8 / (aspect_ratio * pi)) * sqrt(semispan^2 - y^2)
    return CurvedAnalyticWing(semispan, fn)
end

function make_van_dyke_cusped(
    ::Type{CurvedAnalyticWing},
    aspect_ratio :: Real, span :: Real, n :: Int)

    @assert(n >= 0, "N must be postive")
    @assert(n < 7, "Only implemented for n < 7")
    semispan = span / 2
    kns = [1, 4/pi, 3/2, 16/(3 * pi), 15/8, 32/(5*pi), 32/16, 256/(35*pi)]
    kn = kns[n + 1]
    fn = y->2 * kn * semispan * (1 - y^2 / semispan^2)^(n/2) / aspect_ratio
    return CurvedAnalyticWing(semispan, fn)
end

function aspect_ratio(a :: CurvedAnalyticWing)
    wa = area(a)
    return 4 * a.semispan ^ 2 / wa
end

function area(
    a :: CurvedAnalyticWing)

    # Integrate chord from across span.
    return HCubature.hquadrature(a.chord_fn, -a.semispan, a.semispan)[1]
end

"""
The chord of a wing at a point on the span in global coordinates.
"""
function chord(
    a :: CurvedAnalyticWing,
    y :: Real)
    @assert(abs(y) <= a.semispan, "Global coordinate must be in"*
        " [-semispan, semispan].")
    return a.chord_fn(y)
end

function chord(
    a :: CurvedAnalyticWing,
    y :: Array{<:Real})
    return map(x->chord(a, x), y)
end

function sweep_angle(
    a :: CurvedAnalyticWing,
    y :: Real)
    error("To do!")
end

function sweep_angle(
    a :: CurvedAnalyticWing,
    y :: Array{<:Real})
    return map(x->sweep_angle(a, x), y)
end

function radius_of_curvature(
    a :: CurvedAnalyticWing,
    y :: Real)
    error("To do!")
end

function radius_of_curvature(
    a :: CurvedAnalyticWing,
    y :: Array{<:Real})
    return map(x->radius_of_curvature(a, x), y)
end


# END CurvedAnalyticWing.jl
#============================================================================#


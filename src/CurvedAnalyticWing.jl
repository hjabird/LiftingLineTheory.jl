#
# CurvedAnalyticWing.jl
#
# Copyright HJA Bird 2019
#==============================================================================#

import HCubature
import ForwardDiff

mutable struct CurvedAnalyticWing
    semispan :: Real        # Half the full span of the wing
    chord_fn :: Function    # Defined in [-semispan, semispan]
    curve_fn :: Function    # Defined in [-semispan, semispan], +ve is forward

    function CurvedAnalyticWing(
        semispan :: Real, chord_fn :: Function, curve_fn :: Function)

        @assert(hasmethod(chord_fn, (Float64,)), "The chord function is not "*
            "defined for arguments of type Float64")
        @assert(hasmethod(curve_fn, (Float64,)), "The curve function is not "*
            "defined for arguments of type Float64")
        new(semispan, chord_fn, curve_fn)
    end
    function CurvedAnalyticWing(
        wing :: StraightAnalyticWing, curve_fn :: Function)

        @assert(hasmethod(curve_fn, (Float64,)), "The curve function is not "*
            "defined for arguments of type Float64")
        new(wing.semispan, wing.chord_fn, curve_fn)
    end
end

function aspect_ratio(a :: CurvedAnalyticWing)
    wa = area(a)
    return 4 * a.semispan ^ 2 / wa
end

function area(
    a :: CurvedAnalyticWing)

    # Integrate chord from accross span.
    return HCubature.hquadrature(a.chord_fn, -a.semispan, a.semispan)[1]
end

function radius_of_curvature(a :: CurvedAnalyticWing, y :: Real)
    dy1 = x->ForwardDiff.derivative(curve_fn, x)
    dy2 = x->ForwardDiff.derivative(dy1, x)
    rad = abs((1 + dy1(y))^(3/2)) / abs(dy2(y))
    return rad
end

function sweep_angle(a :: CurvedAnalyticWing, y :: Real)
    dy1 = x->ForwardDiff.derivative(curve_fn, x)
    angle = atan(dy1(y))
    return angle
end

# END CurvedAnalyticWing.jl
#============================================================================#

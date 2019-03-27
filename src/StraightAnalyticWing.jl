#
# StraightAnalyticWing.jl
#
# Copyright HJA Bird 2019
#==============================================================================#

import HCubature

mutable struct StraightAnalyticWing
    semispan :: Real        # Half the full span of the wing
    chord_fn :: Function    # Defined in [-semispan, semispan]

    function StraightAnalyticWing(
        semispan :: Real, chord_fn :: Function) 
        new(semispan, chord_fn)
    end
end

"""
    LiftingLineTheory.make_rectangular(type, aspect_ratio, span)

Generate a rectangular wing of given type.
"""
function make_rectangular(
    ::Type{StraightAnalyticWing}, 
    aspect_ratio :: Real, span :: Real ) 

    fn = y -> span / aspect_ratio
    semispan = span / 2
    return StraightAnalyticWing(semispan, fn)
end

"""
    LiftingLineTheory.make_elliptic(type, aspect_ratio, span)

Generate an elliptic wing of given type.
"""
function make_elliptic(
    ::Type{StraightAnalyticWing}, 
    aspect_ratio :: Real, span :: Real )

    semispan = span / 2
    # We multiply by two since we want the sum of the LE + TE offsets.
    fn = y -> (8 / (aspect_ratio * pi)) * sqrt(semispan^2 - y^2)
    return StraightAnalyticWing(semispan, fn)
end

"""
    LiftingLineTheory.make_van_dyke_cusped(type, aspect_ratio, span, cusping_factor)

Generate an wing with cusping according to the formula presented by Van Dyke 
(Lifting-line theory as a singular purturbation problem, 1964, J. App. Math. & Mech.).
Wing is of given type.

cusping factor (int):
    0: rectangular wing
    1: elliptic
    2: lenticular (pointy ends)
    3: cusped 
    4+: even more cusped (max implemented is 7)
"""
function make_van_dyke_cusped(
    ::Type{StraightAnalyticWing},
    aspect_ratio :: Real, span :: Real, n :: Int)

    @assert(n >= 0, "N must be postive")
    @assert(n < 7, "Only implemented for n < 7")
    semispan = span / 2
    kns = [1, 4/pi, 3/2, 16/(3 * pi), 15/8, 32/(5*pi), 32/16, 256/(35*pi)]
    kn = kns[n + 1]
    fn = y->2 * kn * semispan * (1 - y^2 / semispan^2)^(n/2) / aspect_ratio
    return StraightAnalyticWing(semispan, fn)
end

function aspect_ratio(a :: StraightAnalyticWing)
    wa = area(a)
    return 4 * a.semispan ^ 2 / wa
end

function area(
    a :: StraightAnalyticWing)

    # Integrate chord from accross span.
    return HCubature.hquadrature(a.chord_fn, -a.semispan, a.semispan)[1]
end

# END StraightAnalyticWing.jl
#============================================================================#

#
# SteadySweptLLT.jl
#
# Copyright HJA Bird 2019
#==============================================================================#

mutable struct SteadySweptLLT
    wing :: CurvedAnalyticWing
    U :: Real
    aoa_distribution :: Function
end

function Y_to_y(a::SteadySweptLLT, Y::Real)
    return Y / a.wing.semispan
end

function y_to_Y(a::SteadySweptLLT, y::Real)
    return y * a.wing.semispan
end

# PROBABLY WRONG ------------
function Gamma_0(a::SteadySweptLLT, Y::Real)
    chord = a.wing.chord_fn(Y)
    U = a.U
    return chord * U * pi
end

function dGamma_0_dy(a::SteadySweptLLT, Y::Real)
    chord = a.wing.chord_fn(Y)
    U = a.U
    return chord * U * pi
end

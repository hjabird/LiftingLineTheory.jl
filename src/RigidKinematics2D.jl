#
# RigidKinematics2D.jl
#
# Control vertical displacement and pitching about a point.
#
# Copyright HJAB 2019
#
################################################################################

mutable struct RigidKinematics2D
    z_pos :: Function
    dzdt :: Function
    AoA :: Function
    dAoAdt :: Function
    pivot_position :: Real
    function RigidKinematics2D(z::Function, AoA::Function, pivot_position)
        @assert(hasmethod(z, (Float64,)), "z function "*
            "must accept single argument of time.")
        @assert(hasmethod(z, (Real,)), "z function "*
            "must accept real arguments for automatic differentiation.")
        @assert(hasmethod(AoA, (Float64,)),  "AoA function "*
        "must accept single argument of time.")
        @assert(hasmethod(AoA, (Real,)), "AoA function "*
            "must accept real arguments for automatic differentiation.")
        return new(
            z, x->ForwardDiff.derivative(z, x), 
            AoA, x->ForwardDiff.derivative(AoA, x), pivot_position)
    end
end



mutable struct LAULLT
    inner_sols :: LAUTAT
    inner_sol_positions :: Vector{Float64}

    wing_platform :: StraightAnalyticWing

    function LAULLT(;)
        return new()
    end
end

mutable struct IncompleteVortexLattice
    vertices :: Matrix{Float64} # No edge at [i,j] i == 1.
    strengths :: Matrix{Float64}
end

function induced_velocity(a::IncompleteVortexLattice, mes_pnts::Matrix{<:Real})
    @error("TODO!")
    return
end

"""The 3D downwash on the lifting line collocation points."""
function outer_induced_downwash(a::LAULLT)
    @error("TODO!")
    return
end

"""The 2D wake in the outer domain"""
function outer_2D_induced_downwash(a::LAULLT)
    @error("TODO")
    return
end

"""Interpolate 2D wakes to create ring lattice in 3D & 2D points in outer 
domain"""
function construct_wake_lattice(a::LAULLT)
    @error("TODO")
    return
end

function apply_downwash_to_inner_solution(a::LAULLT)
    @error("TODO")
    return
end

function lift_coefficient(a::LAULLT)
    @error("TODO")
    return
end

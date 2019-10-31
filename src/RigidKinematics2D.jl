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

    # Constructor that uses automatic differentiation
    function RigidKinematics2D(z::Function, AoA::Function, pivot_position::Real)
        @assert(hasmethod(z, (Float64,)), "z function "*
            "must accept single Float64 argument of time.")
        @assert(hasmethod(z, (Real,)), "z function "*
            "must accept real arguments for automatic differentiation.")
        @assert(hasmethod(AoA, (Float64,)),  "AoA function "*
        "must accept single Float64 argument of time.")
        @assert(hasmethod(AoA, (Real,)), "AoA function "*
            "must accept real arguments for automatic differentiation.")
        @assert(isfinite(pivot_position), 
            "Denormal value given for pivot position")
        return new(
            z, x->ForwardDiff.derivative(z, x), 
            AoA, x->ForwardDiff.derivative(AoA, x), pivot_position)
    end

    # Constructor that requires explicit derivative definitions
    function RigidKinematics2D(
        z::Function, dzdt::Function,
        AoA::Function, dAoAdt::Function,
        pivot_position::Real)

        @assert(hasmethod(z, (Float64,)), "z function "*
            "must accept single Float64 argument of time.")
        @assert(hasmethod(z, (Float64,)), "dzdt function "*
            "must accept single Float64 argument of time.")
        @assert(hasmethod(AoA, (Float64,)),  "AoA function "*
            "must accept single Float64 argument of time.")
        @assert(hasmethod(AoA, (Float64,)),  "dAoAdt function "*
            "must accept single Float64 argument of time.")
        @assert(isfinite(pivot_position), 
            "Denormal value given for pivot position")
        return new(z, dzdt, AoA,dAoAdt, pivot_position)
    end
end

# Generator functions ----------------------------------------------------------

function make_plunge_function(::Type{RigidKinematics2D}, z::Function)
    @assert(hasmethod(z, (Float64,)), "z function "*
        "must accept single argument of time.")
    @assert(hasmethod(z, (Real,)), "z function "*
        "must accept real arguments for automatic differentiation."*
        " If this isn't possible give the vel explicitly as 3rd argument.")
    return RigidKinematics2D(z, x->0, 0)
end

function make_plunge_function(::Type{RigidKinematics2D}, 
    z::Function, dzdt::Function)

    @assert(hasmethod(z, (Float64,)), "z function "*
        "must accept single argument of time.")
    @assert(hasmethod(dzdt, (Float64,)), "dzdt function "*
        "must accept single argument of time.")
    return RigidKinematics2D(z, dzdt, x->0, x->0, 0)
end

function make_pitch_function(::Type{RigidKinematics2D}, 
    pivot_position::Real, AoA::Function)

    @assert(hasmethod(AoA, (Float64,)),  "AoA function "*
    "must accept single argument of time.")
    @assert(hasmethod(AoA, (Real,)), "AoA function "*
        "must accept real arguments for automatic differentiation."*
        " If this isn't possible give the dAoAdt explicitly as 4th argument.")
    @assert(isfinite(pivot_position), 
        "Denormal value given for pivot position")
    return RigidKinematics2D(x->0, AoA, pivot_position)
end

function make_pitch_function(::Type{RigidKinematics2D}, 
    pivot_position::Real, AoA::Function, dAoAdt::Function)

    @assert(hasmethod(AoA, (Float64,)),  "AoA function "*
        "must accept single argument of time.")
    @assert(hasmethod(dAoAdt, (Float64,)),  "dAoAdt function "*
        "must accept single argument of time.")
    @assert(isfinite(pivot_position), 
        "Denormal value given for pivot position")
    return RigidKinematics2D(x->0, x->0, AoA, dAoAdt, pivot_position)
end

# IO ---------------------------------------------------------------------------

function csv_titles(a::RigidKinematics2D)
    return ["Time" "Z_pos" "Z_vel" "AoA" "dAoAdt" "pivot_location"]
end

function csv_row(a::RigidKinematics2D, t::Float64)
    @assert(isfinite(t))
    return [t, a.z_pos(t), a.dzdt(t), a.AoA(t), a.dAoAdt(t), a.pivot_position]'
end

function csv_row(a::RigidKinematics2D, ts::Vector{Float64})
    ret = mapreduce(t->csv_row(a, t), vcat, ts)
    return ret
end

function from_matrix(::Type{RigidKinematics2D}, mat :: Matrix{<:Real};
        interpolator::Type=CubicSpline{Float64})
    @assert(size(mat)[2] == 6, "Matrix should have size columns: "*
        "{Time Z_pos Z_vel AoA dAoAdt pivot_location}")
    @assert(size(mat)[1] >= 1, "Input matrix should have at least one row.")
    @assert(all(y->y==mat[1,6], mat[:,6]), 
        "The pivot_location (column 6) should remain constant, but it doesn't.")

    @assert(issorted(mat[:,1]),
        "The values of time mat[i,1] should be increasing with respect to i."*
        " Try mat[sortperm(mat[:,1]), :] if you want to do this.")
    @assert(all(i->mat[i,1]!=mat[i+1,1], 1:(size(mat)[1]-1)),
        "Not all time values in the matrix are unique.")

    @assert(interpolator <: Interpolator1D, "Interpolation function must be"*
        " a subtype of LiftingLineTheory.Interpolator1D.")

    # Now we go interpolate stuff...
    z_pos_spl = interpolator(mat[:,1], mat[:, 2])
    z_pos_deriv_spl = interpolator(mat[:,1], mat[:, 3])
    AoA_spl = interpolator(mat[:,1], mat[:, 4])
    AoA_deriv_spl = interpolator(mat[:,1], mat[:, 5])
    
    kinem = RigidKinematics2D(
        t->z_pos_spl(t),
        t->z_pos_deriv_spl(t),
        t->AoA_spl(t),
        t->AoA_deriv_spl(t),
        mat[1,6])
    return kinem
end

# EOF

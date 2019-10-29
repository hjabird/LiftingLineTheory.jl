#
# ThinFoilGeometry.jl
#
# Object to represent a thin aerofoil.
#
# Copyright HJAB 2019
#
################################################################################

mutable struct ThinFoilGeometry
    semichord :: Real
    camber_line :: Function # In [-1, 1]
    camber_slope :: Function

    function ThinFoilGeometry()
        return new(0.5, x->0, x->0)
    end
    function ThinFoilGeometry(semichord::Real)
        @assert(semichord>0, "Semichord must be positive")
        return new(semichord, x->0, x->0)
    end
    function ThinFoilGeometry(semichord::Real, camber_func::Function)
        @assert(semichord>0, "Semichord must be positive")
        @assert(hasmethod(camber_func, (Float64,)), "Camber function "*
            "must accept single argument in [-1 (LE), 1 (TE)].")
        @assert(hasmethod(camber_func, (Real,)), "Camber function "*
            "must accept real arguments for automatic differentiation.")
        return new(semichord, camber_func, x->ForwardDiff.derivative(camber_func, x))
    end
    function ThinFoilGeometry(semichord::Real, camber_func::Function,
        camber_slope::Function)
        @assert(semichord>0, "Semichord must be positive")
        @assert(hasmethod(camber_func, (Float64,)), "Camber function "*
            "must accept single argument in [-1 (LE), 1 (TE)].")
        @assert(hasmethod(camber_slope, (Float64,)), "Camber slope function "*
            "must accept single argument in [-1 (LE), 1 (TE)].")
        return new(semichord, camber_func, camber_slope)
    end
end

function make_flat_plate(
    ::Type{ThinFoilGeometry}, semichord :: Real)

    @assert(semichord > 0, "Semichord must be positive")
    return ThinFoilGeometry(semichord, x->0, x->0)
end

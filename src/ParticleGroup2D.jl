#
# ParticleGroup2D.jl
#
# Represent a group of vortex particles in a way easy to use with CVortex.jl
#
# Copyright HJA Bird 2019-2020
#
################################################################################


mutable struct ParticleGroup2D
    positions :: Matrix{Float32} # An N by 3 Matrix
    vorts :: Vector{Float32}

    function ParticleGroup2D()
        return new(zeros(Float32, 0, 2), zeros(Float32, 0))
    end

end

function total_vorticity(a::ParticleGroup2D)
    return sum(a.vorts)
end

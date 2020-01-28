#
# DownwashEnum.jl
#
# An enumerator to represent the different downwash models for 
# HarmonicULLT.jl
#
# Copyright HJA Bird 2019
#
#==============================================================================#

@enum DownwashModel begin
	# Psuedosteady: no wake "memory" - shed vortex is of const. strength
	# dGamma/dy
    psuedosteady = 1
	
	# StreamwiseFilaments: The vorticity in the streamwise direction is 
	# considered, but spanwise vorticity is neglected
    streamwise_filaments = 2
	
	# Unsteady: The full unsteady wake is considered.
    unsteady = 3
	
	# Strip theory: The wing is modelled as 2D strips - no three dimensional 
	# interaction is considered.
    strip_theory = 4
end

# END DownwashEnum.jl
#=============================================================================#

#
# LiftingLineTheory.jl
#
# Copyright HJA Bird 2019
# h.bird.1@research.gla.ac.uk
# github.com/hjabird
#
#============================================================================#

__precompile__(true)

module LiftingLineTheory

export
    DownwashModel,
	HarmonicULLT,
	StraightAnalyticWing,
	LAULLT,
	
	# For getting results from ULLTs
	bound_vorticity,
	lift_coefficient,
	moment_coefficient,
	compute_fourier_terms!,
	compute_collocation_points!,
	to_vtk,
	
	# For querying wings
	aspect_ratio, 
	area,
	
	# DownwashModel enumerations
	strip_theory,
	unsteady,
	streamwise_filaments,
	psuedosteady
	
using CVortex
import ForwardDiff, WriteVTK
# source files
include("DownwashEnum.jl")			# An enum. No deps.
include("SpecialisedFunctions.jl")	# Wagner etc. + Exponential integral
include("Interpolators1D.jl")		# Cubic spline interpolation
include("TheodorsenSimple.jl")		# Theodorsen + simple evaluation of Thoed.
include("StraightAnalyticWing.jl")	# Define wing  planform shape
include("HarmonicULLT.jl")			# Harmonic ULLT
include("LAUTAT.jl")				# Large amplitude thin aerofoil theory
include("LAULLT.jl")				# Large amplitude lifting line theory
end

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
	StraightAnalyticWing,
    DownwashModel,
	HarmonicULLT,
	LAUTAT,
	LAULLT,
	RigidKinematics2D,
	
	# Simulation control
	advance_one_step,
	compute_fourier_terms!,
	compute_collocation_points!,

	# Post-processing
	bound_vorticity,
	lift_coefficient,
	moment_coefficient,
	lift_and_drag_coefficients,

	# IO
	to_vtk,
	csv_titles,
	csv_row,
	from_matrix,
	
	# For querying wings
	aspect_ratio, 
	area,
	
	# DownwashModel enumerations
	strip_theory,
	unsteady,
	streamwise_filaments,
	psuedosteady,

	# Useful fuctions 
	eldredge_ramp,
	wagner_fn,
	theodorsen_fn,
	sears_fn,

	# Generation of geometries / kinematics
	make_rectangular,
	make_elliptic,
	make_flat_plate,
	make_plunge_function,
	make_pitch_function
	
using CVortex
import ForwardDiff, WriteVTK, FastGaussQuadrature, HCubature
# source files
include("DownwashEnum.jl")			# An enum. No deps.
include("SpecialisedFunctions.jl")	# Wagner etc. + Exponential integral
include("RigidKinematics2D.jl")		# Rigid 2D RigidKinematics2D
include("ThinFoilGeometry.jl")		# Thin aerofoil representation
include("Interpolators1D.jl")		# Cubic spline interpolation
include("TheodorsenSimple.jl")		# Theodorsen + simple evaluation of Thoed.
include("StraightAnalyticWing.jl")	# Define wing  planform shape
include("SteadyLLT.jl")				# Prandtl lifting-line theory
include("HarmonicULLT.jl")			# Harmonic ULLT
include("ParticleGroup2D.jl")		# A vortex particle holder.
include("LAUTAT.jl")				# Large amplitude thin aerofoil theory
include("LAULLT.jl")				# Large amplitude lifting line theory
end #END module

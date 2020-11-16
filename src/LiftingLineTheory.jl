#
# LiftingLineTheory.jl
#
# Copyright HJA Bird 2019-2020
# h.bird.1@research.gla.ac.uk
# github.com/hjabird
#
#============================================================================#

__precompile__(true)

module LiftingLineTheory

export
	CurvedAnalyticWing,
	DownwashModel,
	GuermondUnsteady,
	GuermondUnsteady2,
	HarmonicULLT,
	HarmonicULLT2,
	HarmonicUpwash2D,
	LAULLT,
	LAUTAT,
	LDVM,
	UnsteadyVortexLatticeMethod,
	UnsteadyVortexLatticeMethod2D,
	PEVLM,
	LMPEVLM,
	RigidKinematics2D,
	StraightAnalyticWing,
	SclavounosULLT,
	VortexLattice,
	
	# Simulation control
	advance_one_step!,
	compute_fourier_terms!,
	compute_collocation_points!,

	# Post-processing
	a0_value,
	bound_vorticity,
	lift_coefficient,
	moment_coefficient,
	lift_and_drag_coefficients,

	# IO
	to_vtk,
	csv_titles,
	csv_row,
	from_matrix,
	
	# For querying geometry
	aspect_ratio, 
	area,
	get_vertices,
	set_vertices!,
	to_filaments,
		
	# DownwashModel enumerations
	strip_theory,
	unsteady,
	streamwise_filaments,
	pseudosteady,

	# Useful fuctions 
	eldredge_ramp,
	make_normalised_eldredge_ramp,
	wagner_fn,
	theodorsen_fn,
	sears_fn,

	# Generation of geometries / kinematics
	make_rectangular,
	make_elliptic,
	make_flat_plate,
	make_plunge_function,
	make_pitch_function,
	make_sinusoidal_gust_function
	
	using CVortex
	using LinearAlgebra
	import ForwardDiff, WriteVTK, FastGaussQuadrature, HCubature, QuadGK, FFTW
	# source files
	include("DownwashEnum.jl")			# An enum. No deps.
	include("PointsVortsODE.jl")		# Solve topical ODEs.
	include("SpecialisedFunctions.jl")	# Wagner etc. + Exponential integral
	include("RigidKinematics2D.jl")		# Rigid 2D RigidKinematics2D
	include("ThinFoilGeometry.jl")		# Thin aerofoil representation
	include("Interpolators1D.jl")		# Cubic spline interpolation
	include("TheodorsenSimple.jl")		# Theodorsen + simple evaluation
	include("HarmonicUpwash2D.jl")		# Kussner-Schwarz general solution 
	include("StraightAnalyticWing.jl")	# Define wing planform shape
	include("CurvedAnalyticWing.jl")	# Define wing planform w/ curvature
	include("VortexFilament.jl")		# Vortex filaments convenience methods.
	include("VortexLattice.jl")			# A vortex lattice
	include("ParticleGroup2D.jl")		# A vortex particle holder.
	include("ParticleGroup3D.jl")		# A vortex particle holder.
	include("BufferedParticleWake.jl")	# Lattice that converts to vortons.
	include("SteadyLLT.jl")				# Prandtl lifting-line theory
	include("HarmonicULLT.jl")			# Harmonic ULLT
	include("HarmonicULLT2.jl")			# Harmonic ULLT
	include("SclavounosULLT.jl")		# A faithful reproduction of Sclavounos.
	include("GuermondUnsteady.jl")		# HF Harmonic ULLT - Rect. Wing only.
	include("GuermondUnsteady2.jl")		# HF Harmonic ULLT - Straight wings only.
	include("WingFrequencyResponse.jl") # Multiple HarmonicULLTs at different fqs.
	include("LAUTAT.jl")				# Large amplitude thin aerofoil theory
	include("LDVM.jl")					# LESP modulated discrete vortex method.
	include("LAULLT.jl")				# Large amplitude lifting line theory
	include("UnsteadyVortexLatticeMethod.jl")	# UVLM
	include("UnsteadyVortexLatticeMethod2D.jl")	# UVLM but in 2D
	include("PEVLM.jl")					# Vortex particle enhanced UVLM.
	include("LMPEVLM.jl")				# LESP modulated PEVLM.
end #END module

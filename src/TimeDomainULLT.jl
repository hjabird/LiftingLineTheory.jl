#
# TimeDomainULLT.jl
#
# Copyright HJA Bird 2019
#
#==============================================================================#

import FFTW
import Dierckx

mutable struct TimeDomainULLT
	time_fn :: Function
    free_stream_vel :: Real

    wing :: StraightAnalyticWing
    amplitude_fn :: Function        # Amplitude of oscillation wrt/ span pos.
    pitch_plunge :: Int64           # Plunge = 3, Pitch = 5. Otherwise invalid.

    downwash_model :: DownwashModel # See DownwashModel defintion
	normalised_sample_fq :: Float64	# Samples per span travelled
	normalised_wake_considered :: Float64	# Num wake lengths to include in calc.
    num_terms :: Int64              # Number of terms in fourier expansion
    fourier_terms :: Vector{Real}
    collocation_points :: Vector{Real}  # In terms of theta in [0, pi]
	transfer_fn :: Function			# Frequency domain transfer function

    function TimeDomainULLT(
        time_fn :: Function,
        wing :: StraightAnalyticWing;
        free_stream_vel = 1,
        amplitude_fn = y -> 1,
        pitch_plunge = 3,
        downwash_model = unsteady,
        num_terms = 8,
		normalised_sample_fq = 1000,
		normalised_wake_considered = 10,
        fourier_terms = Vector{Float64}(undef, 1),
        collocation_points = Vector{Float64}(undef, 1),
        transfer_fn = x->error("Transfer function not yet computed. Use"*
            " compute_transfer_function!.")
    )
       @assert(hasmethod(time_fn, (Float64,)))
       @assert(wing.semispan > 0, "Wing must have a positive span")
       @assert(wing.chord_fn(0) >= 0, "Wing must have positive chord")

       new(time_fn, free_stream_vel, wing, amplitude_fn, pitch_plunge,
        downwash_model, normalised_sample_fq, normalised_wake_considered, 
        num_terms, fourier_terms, collocation_points, transfer_fn)
    end
end

function compute_transfer_function!(
    a :: TimeDomainULLT;
    spanwise_interpolation_points=
        collect(-a.wing.semispan*0.9999 : a.wing.semispan/100 : 
        a.wing.semispan*0.9999),
    span_reduced_frequencies= (collect(1 : 0.2 :5)./3).^8 )
    
    srfs = span_reduced_frequencies
    sip = spanwise_interpolation_points
    fqs = srfs .* a.free_stream_vel ./ a.wing.semispan
    tlift_coeffs = Vector{Complex{Float64}}(undef, length(fqs))
    swlift_coeffs = Matrix{Complex{Float64}}(undef, length(fqs), length(sip))
    constcoeff = 1 / a.free_stream_vel

    for i = 1 : length(fqs)
        fq = fqs[i]
        prob = HarmonicULLT(
            fq, a.wing, free_stream_vel=a.free_stream_vel,
            amplitude_fn=a.amplitude_fn, pitch_plunge=a.pitch_plunge,
            downwash_model=a.downwash_model, num_terms=a.num_terms)
        compute_collocation_points!(prob)
        compute_fourier_terms!(prob)
        coeff = im * fq * constcoeff
        # Total coeff:
        tlift_coeffs[i] = lift_coefficient(prob) * coeff
        # And the spanwise:
        try
            swlift_coeffs[i, :] = map(
                y->lift_coefficient(prob, y) * coeff,
                sip)
        catch AmosException
            error("AmosException thrown whilst evaluating lift coeffs wrt/"*
                " span for harmonic ULLT. This might be caused by evaluating"*
                " lift at a chord = 0 wingtip. Consider extrapolating tip"*
                " lift coeff.")
        end
    end
    # Now interpolate to generate fn.
    it1sr = Dierckx.Spline1D(fqs, real.(tlift_coeffs); bc="extrapolate")
    it1si = Dierckx.Spline1D(fqs, imag.(tlift_coeffs); bc="extrapolate")
    it2sr = Dierckx.Spline2D(fqs, sip, real.(swlift_coeffs))
    it2si = Dierckx.Spline2D(fqs, sip, imag.(swlift_coeffs))
    function it(fq :: Real)
       return it1sr(fq) + im * it1si(fq)
    end
    function it(fq :: Real, swp :: Real)
        @assert(-a.wing.semispan <= swp <= a.wing.semispan, "Tried to "*
            "lift function outside of wing.")
        return it2s(fq) + im * it2si(fq)
    end
    a.transfer_fn = it
    return
end

function lift_coefficient(
    a :: TimeDomainULLT, t :: Real; N :: Int = 10)

    # Transfer input function to s-space
    x_s = s->laplace(a.time_fn, s)
    # Multiply through by transfer_fn
    y_s = s->x_s(s) * a.transfer_fn(s)
    # And return to the time domain
    y_t = gaver_stehfest(y_s, t, N)
    return y_t
end

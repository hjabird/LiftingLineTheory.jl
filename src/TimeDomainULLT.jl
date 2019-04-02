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
    transfer_fn :: ExponentialApproximant{Real}	# Frequency domain transfer function

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
        transfer_fn = ExponentialApproximant(Vector{Real}(undef, 1), 
                        Vector{Real}(undef, 1))
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
    spanwise_interpolation_points= a.collocation_points,
    span_reduced_frequencies = (collect(1 : 0.2 :5)./3).^8 )
    
    srfs = [0.00001, 2., 4., 6.] # Go beyond ~8 and F gets dodgy.
    F, fqs, y_pts = generate_f_curves(a, srfs)
    # Now generate the interaction function approx for each y point
    #= form is sum( ( jka_i ) / ( jk - b_i ) ) where j is imag, k is fq var
    and a_i and b_i are coefficient.   
    In time domain this becomes sum( a_i exp(b_i * t))                      =#
    nt = length(srfs)
    as = Matrix{Float64}(undef, nt, length(y_pts))
    bs = Matrix{Float64}(undef, nt, length(y_pts)) 
    for i = 1 : length(y_pts)
        fv = F[:, i]
        # Problem here?
        as[:, i], bs[:, i] = generate_interaction(fv, fqs)
    end
    # Now generate interpolation of our as & bs.
    # We know bs are constant cross span.
    # Use splines and assume symettric.
    a_spl = Vector{Dierckx.Spline1D}(undef, nt)
    b_spl = Vector{Dierckx.Spline1D}(undef, nt)
    for i = 1 : nt
        a_spl[i] = Dierckx.Spline1D(vcat(-y_pts, reverse(y_pts)), 
            vcat(as[i, :], reverse(as[i, :])); bc="extrapolate")
        b_spl[i] = Dierckx.Spline1D(vcat(-y_pts, reverse(y_pts)), 
            vcat(bs[i, :], reverse(bs[i, :])); bc="extrapolate")
    end
    # And use all this to create une function!
    function transfer_fn(t :: Real, y :: Real)
        @assert(t >= 0.0)
        as = map(spl->Dierckx.evaluate(spl, y), a_spl)
        bs = map(spl->Dierckx.evaluate(spl, y), b_spl)
        return mapreduce(x->x[1] * exp(x[2] * t), +, zip(as, bs))
    end
    a.transfer_fn = transfer_fn
    return
end

"""
    Generate the interaction curves at points on the wing span

Returns (F, fqs, y_pts) where:
F is matrix of F interaction fn values at F[i, j] is ith span reduced Frequency
and jth spanwise positions
fqs are frequencies of input span reduced freqyencies
y_pts are the global y positions to take the F values at
"""
function generate_f_curves(
    a :: TimeDomainULLT, srfs)

    # 1) make a harmonic ULLT based on the TD one
    hullt = HarmonicULLT(1, a.wing; free_stream_vel=a.free_stream_vel,
        amplitude_fn = a.amplitude_fn, pitch_plunge=a.pitch_plunge,
        downwash_model=a.downwash_model, num_terms=a.num_terms)
    compute_collocation_points!(hullt)

    # 2) decide the points frequencies and spanwise positions we're collecting
    #   data on.
    fqs = 0.5 .* srfs .* a.free_stream_vel ./ a.wing.semispan
    y_pts = map(x->theta_to_y(hullt, x), hullt.collocation_points)
    F = Matrix{Complex{Float64}}(undef, length(fqs), length(y_pts))

    for i = 1 : length(fqs)
        hullt.angular_fq = fqs[i]
        compute_fourier_terms!(hullt)
        for j = 1 : length(y_pts)
            F[i, j] = f_eq(hullt, y_pts[j])
        end
    end

    return (F, fqs, y_pts)
end

"""
as[:, i], bs[:, i] = generate_interaction(F_values :: Vector{T}, frequencies
    ::Vector)

Compute the a_i and b_i of an approximation of the frequency response of the F
function.
"""
function generate_interaction(F_values :: Vector{T},
    frequencies :: Vector{S}
    ) where {T<:Complex, S<:Real}
    # See HJAB Notes 5 around pg.75

    num_terms = length(frequencies)
    @assert(num_terms >= 2)
    @assert(length(F_values) == length(frequencies))
    as = Vector{Float64}(undef, num_terms)
    bs = Vector{Float64}(undef, num_terms)
    #= 
    Constraints:
    1) We expect as fq->inf, F-> zero, so sum (a_i) = 0
    2) We expect as fq->0, F->F(0) - this is important to us, so enforce as 
    boundary condition. -> b_1 = 0, a_1 = F(0)

    What we want:
    - To select the remaining values of a_i and b_i that gives us the closest 
    approximation of our F.
    Possible methods:
    - Collocation points with predefined b_i. Solve for A_i.
        - How do we choose b_i, and k_j? We have to preselect fqs. Based on 
        the expected shape of our F curves - ie, srfs below 8 are important. 
        Very important below 4. Do we use real or imag?
    - Some other kind of method?
        - We calculating lots of F points ain't hard...

    Collocation method:
    =#
    bs[1] = 0.;
    as[1] = real(F_values[1])
    # Choose collocations fqs and b values
    fq_idxs = collect(2 : num_terms)
    bs[2:end] = -frequencies[fq_idxs]
    ks = frequencies[fq_idxs[1:end-1]]
    # Assemble matrix
    mat = Matrix{Float64}(undef, num_terms-1, num_terms-1)
    # Constraint that our a_i must sum to zero
    mat[end, :] .= 1.
    # Expressions for F = sum...
    mat[1:end-1, :] = map(
        i->ks[i[1]]^2 / (ks[i[1]]^2 + bs[i[2]]^2),
        collect((i, j) for i in 1 : num_terms-2, j in 1 : num_terms-1)
    )
    rhs = Vector{Float64}(undef, num_terms-1)
    rhs[1:end-1] = real.(F_values[fq_idxs[1:end-1]])
    rhs[end] = 0.
    rhs .-= as[1]
    # Now we can solve:
    as[2:end] = mat \ rhs
    # And with luck we have a reasonable approximation of the input.
    return as, bs
end

function lift_coefficient(
    a :: TimeDomainULLT, t :: Real)



end

#
# TimeDomainULLT.jl
#
# Copyright HJA Bird 2019
#
#==============================================================================#

import FastGaussQuadrature
import ForwardDiff
import NLopt
import PyPlot

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
    transfer_fn_interp :: FDTDMixedApproxInterp	# Frequency domain transfer function

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
        collocation_points = Vector{Float64}(undef, 1)
    )
        @assert(hasmethod(time_fn, (Float64,)))
        @assert(wing.semispan > 0, "Wing must have a positive span")
        @assert(wing.chord_fn(0) >= 0, "Wing must have positive chord")


        transfer_fn_interp = FDTDMixedApproxInterp(
            wing.semispan * [-0.99, 0.99],
            Vector{FDTDMixedApprox}(
               [FDTDMixedApprox(
                    FDTDExpApprox([0.], [0.]),
                    FDTDTExpApprox([0.], [-1.])),
                FDTDMixedApprox(
                    FDTDExpApprox([0.], [0.]),
                    FDTDTExpApprox([0.], [-1.]))]),
            wing.semispan
        )
        new(time_fn, free_stream_vel, wing, amplitude_fn, pitch_plunge,
        downwash_model, normalised_sample_fq, normalised_wake_considered,
        num_terms, fourier_terms, collocation_points, transfer_fn_interp)
    end
end

function compute_transfer_function!(
    a :: TimeDomainULLT;
    spanwise_interpolation_points= a.collocation_points,
    span_reduced_frequencies = (collect(1 : 0.2 :5)./3).^8 )

    srfs = [0.00001, 0.25, 0.5, 1., 2., 3., 5., 8.] # Go beyond ~8 and F gets dodgy.
    F, fqs, y_pts = generate_f_curves(a, srfs)
    #= Approximation:
    We are approximating as Sum(a_i * exp(b_i * t)) + Sum(c_i * t * exp(d_i *t))
    in the time domain.  =#
    nt_exp = 8 #Int64(ceil(length(srfs))/2)        # The exp(b_i t) terms
    nt_texp = 0 #Int64(length(srfs)-nt_exp)        # The t * exp(d_i t) terms
    as = Matrix{Float64}(undef, nt_exp, length(y_pts))  # The a_is
    bs = Matrix{Float64}(undef, nt_exp, length(y_pts))  # The b_is
    cs = Matrix{Float64}(undef, nt_texp, length(y_pts))  # The c_is
    ds = Matrix{Float64}(undef, nt_texp, length(y_pts))  # The d_is
    for i = 1 : length(y_pts)
        fv = F[:, i]
        as[:, i], bs[:, i], cs[:, i], ds[:, i] =
            generate_interaction(fv, fqs, nt_exp, nt_texp)
    end
    # And use all this to create une function!
    y_pts = vcat(-y_pts, reverse(y_pts))
    as = hcat(as, reverse(as; dims=2))
    bs = hcat(bs, reverse(bs; dims=2))
    cs = hcat(cs, reverse(cs; dims=2))
    ds = hcat(ds, reverse(ds; dims=2))
    exp_interp = FDTDExpApproxInterp(y_pts,
        map(i->FDTDExpApprox( as[:, i], bs[:, i]), 1 : length(y_pts)),
        a.wing.semispan)
    texp_interp = FDTDTExpApproxInterp(y_pts,
        map(i->FDTDTExpApprox(cs[:, i], ds[:, i]), 1 : length(y_pts)),
        a.wing.semispan)
    interp_mixed = FDTDMixedApproxInterp(exp_interp, texp_interp)
    a.transfer_fn_interp = interp_mixed
    plot_transfer_fn_against_col(a.transfer_fn_interp, F, y_pts, fqs)
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

Compute the a_i, b_i, c_j & d_j of an approximation of the frequency response of
the F function.
"""
function generate_interaction(F_values :: Vector{T},
    frequencies :: Vector{S},
    nt_exp :: Int,  # Number of a_i * exp(b_i * t) terms
    nt_texp :: Int  # Number of c_i * t * exp(d_i * t) terms
    ) where {T<:Complex, S<:Real}
    # See HJAB Notes 5 around pg.75

    @assert(length(F_values) == length(frequencies))
    @assert(nt_exp > 0, "Number of a_i * exp(b_i * t) terms must be more "*
        "than 0.")
    @assert(nt_texp >= 0, "Number of c_i * t * exp(d_i * t) terms must be 0"*
        " or more than 0.")
    as = Vector{Float64}(undef, nt_exp)
    bs = Vector{Float64}(undef, nt_exp)
    cs = Vector{Float64}(undef, nt_texp)
    ds = Vector{Float64}(undef, nt_texp)
    #=
    Constraints:
    1) We expect as fq->inf, F-> zero, so sum (a_i) = 0
    2) We expect as fq->0, F->F(0) - this is important to us, so enforce as
    boundary condition. -> b_1 = 0, a_1 = F(0)
    =#
    # Initial values:
    bs[1] = 0.;
    bs[2:end] = -frequencies[2:length(bs)]
    as[1] = real(F_values[1])
    # Lets not guess for the other values
    cs = zero(cs)
    ds = [Float64(-i) for i in 1 : length(ds)]
    as, cs = real_collocation(F_values, frequencies, bs, ds, as[1])

    function minobj(x, grad)
        b_i = vcat(bs[1], x[1:nt_exp-1])
        d_i = x[nt_exp:end]
        a_i, c_i = real_collocation(F_values, frequencies, b_i, d_i, as[1])
        if length(grad) > 0
            grad[:] = ForwardDiff.gradient(y->sum(abs.(approximation_collocation_error(
                F_values, frequencies, a_i[1], vcat(bs[1], y[1:nt_exp-1]), y[nt_exp:nt_exp+nt_texp-1])).^2), x)
        end
        errs = approximation_collocation_error(
            F_values, frequencies, a_i[1], b_i, d_i)
        return sum(abs.(errs).^2)
    end

    gradient =  vcat(bs[2:end],ds)
    minobj(vcat(bs[2:end], ds), gradient)
	# LD_MMA, LD_SQP, LD_LBFGS
    opt = NLopt.Opt(:LD_LBFGS, length(bs)-1+length(ds))
    opt.min_objective = minobj
    opt.upper_bounds = -1e-16
    opt.ftol_rel = 1e-16
    opt.maxtime = 1

    #(~, xmin, reason) = NLopt.optimize!(opt, vcat(bs[2:end], ds))
    #println(opt.numevals)
    #println(reason)
    #bs[2:end] = xmin[1:length(bs)-1]
    #ds[:] = xmin[length(bs):end]
    as, cs = real_collocation(F_values, frequencies, bs, ds, as[1])

    println("RESULT")
    println("\tAs: ", as, "+(", sum(as), ")")
    println("\tBs: ", bs)
    println("\tCs: ", cs)
    println("\tDs: ", ds)

    # And with luck we have a reasonable approximation of the input.
    return as, bs, cs, ds
end

function real_collocation(
    Fs :: Vector{A},
    freqs :: Vector{B},
    b_i :: Vector{C},
    d_i :: Vector{D},
    a_1 ::Real
    ) where {A<:Complex, B<:Real, C<:Real, D<:Real}
    #=
    Constraints:
    1) We expect as fq->inf, F-> zero, so sum (a_i) = 0
    2) We expect as fq->0, F->F(0) - this is important to us, so enforce as
    boundary condition. -> b_1 = 0, a_1 = F(0)
    =#

    @assert(length(Fs) == length(freqs), "The list of complex downwashes and "*
        "frequencies at which they were generated do not match length")
    @assert(length(b_i) > 0, "The b_i approximatoin must have more than 0"*
        " terms. There are ", length(b_i), " terms.")
    @assert(b_i[1] == 0.0, "b_i[1] must equal zero due to low freqency limit.")
    @assert(allunique(b_i), "Values of b_i in sum(a_i * fn(b_i, omega)) must"*
        " be unique. Values are "*string(b_i))
    @assert(allunique(d_i), "Values of d_i in sum(a_i * fn(b_i, omega)) must"*
        " be unique. Values are ", d_i)
    @assert(allunique(freqs), "All collocation frequencies must be unique.")

    # Choose collocations fqs and b values
    ks = freqs[2:end-1]
    # Assemble matrix
    mat = real_approximation_matrix(ks, b_i, d_i)[:, 2:end]
    hf_row = hcat(  ones(typeof(mat[1,1]), 1, length(b_i)-1),
                    zeros(typeof(mat[1,1]), 1, length(d_i)))
    mat = vcat(mat, hf_row)
    rhs = real.(Fs[2:end-1])
    rhs = vcat(rhs, 0)
    rhs .-= a_1
    # Now we can solve:
    lhs = rhs
    try
        lhs = mat \ rhs
    catch
        display(mat)
        throw
    end
    as = vcat(a_1, lhs[1:length(b_i)-1])
    cs = lhs[length(b_i):end]
    return as, cs
end

function approximation_collocation_error(
    Fs :: Vector{A},
    freqs :: Vector{B},
    a_i ::Vector{D},
    b_i :: Vector{C},
    c_i :: Vector{E},
    d_i :: Vector{F}
    ) where {A<:Complex, B<:Real, C<:Real, D<:Real, E<:Real, F<:Real}
    @assert(length(Fs) == length(freqs), "The list of complex downwashes and "*
        "frequencies at which they were generated do not match length")
    @assert(b_i[1] == 0, "b_i[1] must equal zero due to low freqency limit.")
    @assert(length(a_i)==length(b_i), "a_i and b_i should have same number of terms.")
    @assert(length(c_i)==length(d_i), "c_i and d_i should have same number of terms.")

    mat = imag_approximation_matrix(freqs, b_i, d_i)
    approx = mat * vcat(a_i, c_i)
    imag_err = imag.(Fs) - approx
    mat = real_approximation_matrix(freqs, b_i, d_i)
    approx = mat * vcat(a_i, c_i)
    real_err = real.(Fs) - approx
    return map(x->x[1] + im*x[2], zip(real_err, imag_err))
end

function approximation_collocation_error(
    Fs :: Vector{A},
    freqs :: Vector{B},
    a_1 :: Real,
    b_i :: Vector{C},
    d_i :: Vector{F}
    ) where {A<:Complex, B<:Real, C<:Real, F<:Real}
    @assert(length(Fs) == length(freqs), "The list of complex downwashes and "*
        "frequencies at which they were generated do not match length")
    @assert(b_i[1] == 0, "b_i[1] must equal zero due to low freqency limit.")

    a_i, c_i = real_collocation(Fs, freqs, b_i, d_i, a_1)
    mat = imag_approximation_matrix(freqs, b_i, d_i)
    approx = mat * vcat(a_i, c_i)
    imag_err = imag.(Fs) - approx
    mat = real_approximation_matrix(freqs, b_i, d_i)
    approx = mat * vcat(a_i, c_i)
    real_err = real.(Fs) - approx
    return map(x->x[1] + im*x[2], zip(real_err, imag_err))
end

function real_approximation_matrix(
    k_i :: Vector{S},
    b_i :: Vector{U},
    d_i :: Vector{V}) where {S<:Real, U<:Real, V<:Real}

    @assert(all(b_i .<= 0), "b_i values must be negative or 0")
    @assert(all(k_i .>= 0), "Frequencies must be positive")
    @assert(all(d_i .< 0), "d_i must be less than 0")

    nt_exp = length(b_i)
    nt_texp = length(d_i)
    mat = zeros(U, length(k_i), nt_exp + nt_texp)

    mat[:, 1:nt_exp] = map(
        i->k_i[i[1]]^2 / (k_i[i[1]]^2 + b_i[i[2]]^2),
        collect((i, j) for i in 1 : length(k_i), j in 1 : nt_exp)
    )
    mat[:, nt_exp + 1:nt_exp + nt_texp] = map(
        i->-2*k_i[i[1]]^2*d_i[i[2]] / (k_i[i[1]]^2 + d_i[i[2]]^2)^2,
        collect((i, j) for i in 1 : length(k_i), j in 1 : nt_texp)
    )
    return mat
end

function imag_approximation_matrix(
    k_i :: Vector{S},
    b_i :: Vector{U},
    d_i :: Vector{V}) where {S<:Real, U<:Real, V<:Real}

    @assert(all(b_i .<= 0), "b_i values must be negative or 0")
    @assert(all(k_i .>= 0), "Frequencies must be positive")
    @assert(all(d_i .< 0), "d_i must be less than 0")

    nt_exp = length(b_i)
    nt_texp = length(d_i)
    mat = zeros(U, length(k_i), nt_exp + nt_texp)

    mat[:, 1:nt_exp] = map(
        i->-b_i[i[2]] * k_i[i[1]] / (k_i[i[1]]^2 + b_i[i[2]]^2),
        collect((i, j) for i in 1 : length(k_i), j in 1 : nt_exp)
    )
    mat[:, nt_exp + 1:nt_exp+nt_texp] = map(
        i->k_i[i[1]]*(d_i[i[2]]^2-k_i[i[1]]^2) / (k_i[i[1]]^2 + d_i[i[2]]^2)^2,
        collect((i, j) for i in 1 : length(k_i), j in 1 : nt_texp)
    )
    return mat
end

function lift_coefficient(
    a :: TimeDomainULLT, t :: Real, dt :: Real)
    @assert(dt > 0)

    cl = mapreduce(
	        ti->lift_coefficient_step(a, t-ti) *
	            ForwardDiff.derivative(a.time_fn, ti) * dt,
	        +,
	        0 : dt : t; init=0.0)
    return cl
end

function lift_coefficient(
    a :: TimeDomainULLT, t :: Real, dt :: Real, y :: Real)
    @assert(abs(x) < a.wing.semispan)
    @assert(dt > 0)

    result = mapreduce(
        ti->lift_coefficient_step(a, t-ti, y) *
            ForwardDiff.derivative(a.time_fn, ti) * dt,
        +,
        0 : dt : t; init=0.0)
    return result
end

function lift_coefficient_step(
    a :: TimeDomainULLT, t :: Real)

    function integrand(y)
        return a.wing.chord_fn(y) * lift_coefficient_step(a, t, y)
    end
    w_area = area(a.wing)
    if t > 0
        nodes, weights = FastGaussQuadrature.gausslegendre(40)
        pts = map(
            x->linear_remap(x[1], x[2], -1, 1, -a.wing.semispan, a.wing.semispan),
            zip(nodes, weights))
        integral = sum(last.(pts) .* map(integrand, first.(pts)))/ w_area
        res = integral
    else
        res = 0
    end
    return res
end

function lift_coefficient_step(
    a :: TimeDomainULLT, t :: Real, x :: Real)
    @assert(abs(x) < a.wing.semispan)
    dwash = interpolate(a.transfer_fn_interp, x)
    wagner = FDTDExpApprox([1., -0.165, -0.335], [0., -0.0455, -0.3])
    if t > 0
        res = td_eval(wagner, t) - duhamel_int(wagner, dwash, t)
    else
        res = 0
    end
    return res;
end

function plot_transfer_fn_against_col(a, F, y_pts, fqs)
    colour = "mrbgkcy"
    for i = 1 : Int64(floor(length(y_pts)/2))
        yp = y_pts[i]
        cidx = i%6 + 1
        dwash = interpolate(a, yp)
        dwashes = map(f->fd_eval(dwash, f), fqs)
        PyPlot.plot(real.(dwashes), imag.(dwashes), colour[cidx]*"o")
        dwashesCont = map(f->fd_eval(dwash, f), vcat(0.00001:0.01:20, 21:1000))
        PyPlot.plot(real.(dwashesCont), imag.(dwashesCont) , colour[cidx]*"-")
        PyPlot.plot(real.(F[:, i]), imag.(F[:, i]), colour[cidx]*"x")
    end
end

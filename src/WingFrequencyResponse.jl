#
# WingFrequencyResponse.jl
#
# The frequency response of a wing obtained using lifting-line theory.
#
# Copyright HJAB 2020
#
################################################################################

mutable struct WingFrequencyResponse
    wing :: StraightAnalyticWing
    pitch_plunge :: Int
    wake_model :: DownwashModel
    free_stream_vel :: Real
    amplitude_fn :: Function # Amplitude of oscillation wrt/ span pos.
    num_terms
    solutions :: Vector{HarmonicULLT}

    function WingFrequencyResponse(
        wing :: StraightAnalyticWing;
        pitch_plunge::Int=3, # 3 = plunge, 5 = midchord pitch
        wake_model::DownwashModel=unsteady,
        free_stream_vel::Real=1,
        amplitude_fn::Function=y->1,
        num_terms::Int=8,
        solutions::Vector{HarmonicULLT}=Vector{HarmonicULLT}()
    ) :: WingFrequencyResponse

        @assert(free_stream_vel > 0)
        @assert(hasmethod(amplitude_fn, (Float64,)))
        @assert(num_terms >= 1)
        @assert(pitch_plunge == 3 || pitch_plunge == 5)
        return new(wing, pitch_plunge, 
            wake_model, free_stream_vel,
            amplitude_fn, num_terms, solutions)
    end
end

function time_domain_respose(
    fn::Function, # Function in time domain
    t0::Real, 
    tend::Real,
    wing_response::WingFrequencyResponse;
    nsamples::Int=1024) :: Tuple{Vector{Float64}, Vector{Float64}}

    @assert(hasmethod(fn, (Float64,)))
    @assert(t0 < tend)
    @assert(nsamples > 1)
    check(wing_response)
    @assert(length(wing_response.solutions) > 1)

    dt = (tend - t0) / (nsamples - 1)
    ts = range(t0, tend; length = nsamples)
    eval_fn = fn.(ts)
    fftres = FFTW.fft(eval_fn)
    fqs = FFTW.fftfreq(nsamples, 1/dt)
    # Convolution in frequency domain:
    omega_fqs = 2 * pi .* fqs
    function lift_fn(omega::Float64)
        cl = lift_coefficient(wing_response, abs(omega))
        cl = omega > 0 ? cl : conj(cl)
        return cl
    end
    lift_coeffs = lift_fn.(omega_fqs)
    conv = lift_coeffs .* fftres
    res = FFTW.ifft(conv)
    return (real.(res), ts)
end

function add_frequency!(a::WingFrequencyResponse, angular_fq::Real) :: Nothing
    prob = create_solution_internal!(a, angular_fq)
    @assert(all(isfinite.(prob.fourier_terms)))
    push!(a.solutions, prob)
    sort_solutions_by_fq!(a)
    return
end

function add_frequency!(a::WingFrequencyResponse, angular_fqs::Vector{<:Real}) :: Nothing
    @assert(all(isfinite.(angular_fqs)))
    for i = 1 : length(angular_fqs)
        prob = create_solution_internal!(a, angular_fqs[i])
        push!(a.solutions, prob)
    end
    sort_solutions_by_fq!(a)
    return
end

function create_solution_internal!(
    a::WingFrequencyResponse, angular_fq::Real) :: HarmonicULLT
    prob = HarmonicULLT(
        angular_fq, a.wing;
        free_stream_vel=a.free_stream_vel,
        amplitude_fn=a.amplitude_fn,
        pitch_plunge=a.pitch_plunge,
        downwash_model=a.wake_model,
        num_terms=a.num_terms)
    
    compute_collocation_points!(prob)
    compute_fourier_terms!(prob)
    return prob
end

function sort_solutions_by_fq!(a::WingFrequencyResponse)
    array = a.solutions
    byfn=x->x.angular_fq
    sort!(array; by=byfn)
    a.solutions = array
    return
end

function frequencies(a::WingFrequencyResponse) :: Vector{Float64}
    fqs = map(x->x.angular_fq, a.solutions)
    return fqs
end

function lift_coefficients(
        a::WingFrequencyResponse) :: Vector{Complex{Float64}}
    check(a)
    cls = map(x->lift_coefficient(x), a.solutions)
    return cls
end

function lift_coefficient(
    a::WingFrequencyResponse, fq::Real; extrapolate::Bool=true) :: ComplexF64
    boundl, boundu, ~ = get_interpolation_basis(a, fq)
    coeff = im * fq / a.free_stream_vel
    df = boundu.angular_fq - boundl.angular_fq
    ca = (fq - boundl.angular_fq) / df
    cb = (boundu.angular_fq - fq) / df
    res = (cb * lift_coefficient(boundl) 
        + ca * lift_coefficient(boundu)) * coeff
    return res
end

function get_interpolation_basis(
    a::WingFrequencyResponse, fq::Real; extrapolate::Bool=true) 

    check(a)
    @assert(length(a.solutions) > 1)
    len = length(a.solutions)
    i = 1;
    extrapolating = 0
    while (i <= len) && (a.solutions[i].angular_fq <= fq)
        i += 1
    end
    if i > len
        if fq == a.solutions[end].angular_fq
            i -= 1
        else
            extrapolating = 1
        end
    elseif fq < a.solutions[1].angular_fq
        extrapolating = -1
    end
    if extrapolating != 0 && !extrapolate
        @assert("Refusing to extrapolate")
    end
    boundl = 0
    bouldu = 0
    if extrapolating == 0 
        boundl = a.solutions[i-1]
        boundu = a.solutions[i]
    elseif extrapolating == -1
        boundl = a.solutions[1] 
        boundu = a.solutions[2]
    elseif extrapolating == 1
        boundl = a.solutions[end-1]
        boundu = a.solutions[end] 
    end
    return boundl, boundu, extrapolating
end

function check(a::WingFrequencyResponse)
    @assert(a.free_stream_vel > 0)
    @assert(hasmethod(a.amplitude_fn, (Float64,)))
    @assert(a.num_terms >= 1)
    @assert(a.pitch_plunge == 3 || pitch_plunge == 5)
    byfn = x->x.angular_fq
    @assert(issorted(a.solutions; by=byfn))
    return
end



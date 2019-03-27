
import SpecialFunctions
import FastGaussQuadrature
import PyPlot
import LinearAlgebra
import Optim

function create_wagner_fn2(num_terms :: Int, k_max :: Real)
    @assert(num_terms > 0)
    I = num_terms
    lim_k_to_inf = 0.5  # Limit as k->infinity for theodorsen_fn
    lim_k_to_zero = 1   # for theodorsen fn.

    ai = Vector{Float64}(undef, I)
    bi = Vector{Float64}(undef, I)
    bi[1] = 0       # Required to satisfy k->0 limit
    ai[1] = lim_k_to_zero
    # Arbitrarily set
    kn = collect( i / (I-2) for i = 1 : I-2)
    # Imaginary
    bi[2:end] = - collect(1:I-1) * k_max / (I - 1)
    function genmat(big)
        @assert(length(kn) == length(big)-1)
        mat_inp = collect((b, k) for k in kn, b in big)
        println("bi: ", big)
        matrix = vcat(
            map(x->-x[1]*x[2]/(x[1]^2 + x[2]^2),
                mat_inp),
            ones(I-1)')
        return matrix
    end
    bi[2:end] = Optim.minimizer(Optim.optimize(x->LinearAlgebra.cond(genmat(x)), bi[2:end]; autodiff = :forward))
    println(bi)
    matrix = genmat(bi[2:end])

    rhs_vec = vcat(imag.(theodorsen_fn.(kn)), lim_k_to_inf)
    a1_vec = vcat(zeros(I-2), ai[1])
    println("kn = \t\t", kn)
    println("bi = \t\t", bi)
    println("A1 = \t\t", a1_vec)
    println("rsh= \t\t", rhs_vec)
    println("mat= \t\t", matrix)
    println("cond(mat) = ", LinearAlgebra.cond(matrix))

    ai[2:end] = matrix \ (rhs_vec - a1_vec)
    function wagner_fn(s :: Real)
        if s >= 0
            return mapreduce(
                x -> x[1] * exp(x[2] * s),
                +,
                zip(ai, bi) )
        else
            return 0
        end
    end
    println("Ai = \t", ai)
    println("Bi = \t", bi)

    ks = collect(0.001: 0.002 : 4)
    tr = real.(theodorsen_fn.(ks))
    ti = imag.(theodorsen_fn.(ks))
    fr = real.(map(k->mapreduce(x->im*k*x[1]/(im*k-x[2]), +, zip(ai, bi)), ks))
    fi = imag.(map(k->mapreduce(x->im*k*x[1]/(im*k-x[2]), +, zip(ai, bi)), ks))
    trd = real.(theodorsen_fn.(kn))
    tid = imag.(theodorsen_fn.(kn))
    frd = real.(map(k->mapreduce(x->im*k*x[1]/(im*k-x[2]), +, zip(ai, bi)), kn))
    fid = imag.(map(k->mapreduce(x->im*k*x[1]/(im*k-x[2]), +, zip(ai, bi)), kn))
    PyPlot.figure()
    PyPlot.plot(tr, ti, "k-")
    PyPlot.plot(fr, fi, "r-")
    println(trd)
    PyPlot.plot(trd, tid, "kx")
    PyPlot.plot(frd, fid, "rx")

    return wagner_fn
end

"""
    theodorsen_fn(k::Real)

Theodorsen's function C(k), where k is chord reduced frequency = omega c / 2 U.
"""
function theodorsen_fn(k :: Real)
    @assert(k >= 0, "Chord reduced frequency should be positive.")

    h21 = SpecialFunctions.hankelh2(1, k)
    h20 = SpecialFunctions.hankelh2(0, k)
    return h21 / (h21 + im * h20)
end

#
# TestTheodorsen.jl
#
# Test the Harmonic Upwash version of Theodorsen's theory.
#
# Copyright HJA Bird 2020
#
################################################################################

using LiftingLineTheory
using SpecialFunctions
using PyPlot

let # Firstly for HEAVE
    println("Heaving:")
    ks = collect(0.01:0.01:5)
    # Adapted from Fung.
    Ck = k->hankelh2(1, k) / (hankelh2(1, k) + im * hankelh2(0, k))
    L_fn = (k, b, ym, U, rho)-> pi * rho * U^2 * ym * k^2 * (1 - 2*im * Ck(k)/k)
    M_fn = (k, b, ym, U, rho)->-pi * rho * U^2 * im *  ym * k * Ck(k)
    cl_fn = (k, b, ym)->L_fn(k, b, ym, 1, 1) / (b)
    #cm_fn = (k, b, ym)-> - im * pi * k * Ck(k) * ym * b/ 2 
    cm_fn = (k, b, ym)-> M_fn(k, b, ym, 1, 1) / (0.5 * (2 * b)^2)
    cls = zeros(ComplexF64, length(ks))
    cms = zeros(ComplexF64, length(ks))
    rcls = zeros(ComplexF64, length(ks))
    rcms = zeros(ComplexF64, length(ks))
    for i = 1:length(ks)
        amp = 0.44
        semichord = 1
        uw = make_plunge_function(HarmonicUpwash2D, amp, ks[i]; semichord=semichord)
        cls[i] = lift_coefficient(uw)
        cms[i] = moment_coefficient(uw)
        rcls[i] = cl_fn(ks[i], semichord, amp)
        rcms[i] = cm_fn(ks[i], semichord, amp)
    end
    clerr = (cls - rcls) ./ rcls
    cmerr = (cms - rcms) ./ rcms
    if any(abs.(clerr) .> 1e-6)
        println("ERROR: Bad cl results!")
    end
    if any(abs.(cmerr) .> 1e-6)
        println("ERROR: Bad cm results! :  average error is ~"*string(sum(abs.(cmerr))/length(ks)))
        figure()
        plot(ks, abs.(cmerr), label="Abs err.")
        plot(ks, real.(cmerr), label="Real err.")
        plot(ks, imag.(cmerr), label="Imag. err.")
        legend()
        title("Plunging moment abs(error)")
    end
end

let # Secondly for PITCH about midchord
    println("Pitching:")
    ks = collect(0.01:0.01:5)
    Ck = k->hankelh2(1, k) / (hankelh2(1, k) + im * hankelh2(0, k))
    cl_fn = (k, b, ym)->2 * pi * ((1+im * k / 2)* Ck(k) + im * k / 2) * ym
    cm_fn = (k, b, ym)->pi / 2 * ((1-im * k / 2)* Ck(k) + im * k / 2 - k^2/8) * ym
    cls = zeros(ComplexF64, length(ks))
    cms = zeros(ComplexF64, length(ks))
    rcls = zeros(ComplexF64, length(ks))
    rcms = zeros(ComplexF64, length(ks))
    for i = 1:length(ks)
        amp = 0.44
        semichord = 1
        uw = make_pitch_function(HarmonicUpwash2D, amp, ks[i]; semichord=semichord)
        cls[i] = lift_coefficient(uw)
        cms[i] = moment_coefficient(uw)
        rcls[i] = cl_fn(ks[i], semichord, amp)
        rcms[i] = cm_fn(ks[i], semichord, amp)
    end
    clerr = (cls - rcls) ./ rcls
    cmerr = (cms - rcms) ./ rcms
    if any(abs.(clerr) .> 1e-6)
        println("ERROR: Bad cl results! :  average error is ~"*string(sum(abs.(clerr))/length(ks)))
    end
    if any(abs.(cmerr) .> 1e-6)
        println("ERROR: Bad cm results! :  average error is ~"*string(sum(abs.(cmerr))/length(ks)))
        figure()
        plot(ks, abs.(cmerr), label="Abs err.")
        plot(ks, real.(cmerr), label="Real err.")
        plot(ks, imag.(cmerr), label="Imag. err.")
        legend()
        title("Pitching moment abs(error)")
    end
end

let # Secondly for ocillating gust
    println("Oscillating gust:")
    ks = collect(0.01:0.01:5)
    Ck = k->hankelh2(1, k) / (hankelh2(1, k) + im * hankelh2(0, k))
    Sears = k->(besselj(0, k) - im * besselj(1, k)) * Ck(k) + im * besselj(1, k)
    L_fn = (k, b, amp)->pi * 2 * b * amp * Sears(k)
    M_fn = (k, b, amp)->L_fn(k,b,amp)  * b / 2
    cl_fn = (k, b, amp)->L_fn(k, b, amp) / b
    cm_fn = (k, b, amp)->M_fn(k, b, amp) / (0.5 * (2*b)^2)
    cls = zeros(ComplexF64, length(ks))
    cms = zeros(ComplexF64, length(ks))
    rcls = zeros(ComplexF64, length(ks))
    rcms = zeros(ComplexF64, length(ks))
    for i = 1:length(ks)
        amp = 0.44
        semichord = 2.35
        uw = make_sinusoidal_gust_function(HarmonicUpwash2D, amp, ks[i]; semichord=semichord)
        cls[i] = lift_coefficient(uw)
        cms[i] = moment_coefficient(uw)
        rcls[i] = cl_fn(ks[i], semichord, amp)
        rcms[i] = cm_fn(ks[i], semichord, amp)
    end
    clerr = (cls - rcls) ./ rcls
    cmerr = (cms - rcms) ./ rcms
    if any(abs.(clerr) .> 1e-6)
        println("ERROR: Bad cl results! :  average error is ~"*string(sum(abs.(clerr))/length(ks)))
    end
    if any(abs.(cmerr) .> 1e-6)
        println("ERROR: Bad cm results! :  average error is ~"*string(sum(abs.(cmerr))/length(ks)))
        plot(ks, abs.(cmerr), label="Abs err.")
        plot(ks, real.(cmerr), label="Real err.")
        plot(ks, imag.(cmerr), label="Imag. err.")
        legend()
    end
end

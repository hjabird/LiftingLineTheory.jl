#
# ThoedorsenSimple.jl
#
# An easy use of thoedorsen's method.
#
# Copyright HJA Bird 2019-2020
#
################################################################################

import SpecialFunctions
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

"""
Evaluate a simple Theodorsen like problem.
pitch_loc maps -1 to LE, 1 to TE.
"""
function theodorsen_simple_cl(k::Real, heave_amp::Number, pitch_amp::Number; 
    pitch_loc::Real=0, chord::Real=1, U::Real=1)
    @assert(k > 0, "Chord reduced frequency must be positive.")
    @assert(pitch_loc == pitch_loc, "Pitch location must be a normal number.")
    @assert(chord > 0, "Chord must be positive")
    @assert(U > 0)
    ck = theodorsen_fn(k)
    h0s = heave_amp / chord;
    alpha0 = pitch_amp
    xp2 = pitch_loc / 2 + 0.5

    clh = 2 * pi * h0s * (k^2 - 2 * im * k * ck)
    cla = 2 * pi * alpha0 * (
        ck * (1 - 2 * im * k * (xp2 - 3 / 4)) 
        + im * k / 2 + k^2 * (xp2 - 1/2))

    return clh + cla
end

"""
Evaluate a simple Theodorsen like problem.
pitch_loc maps -1 to LE, 1 to TE.
"""
function theodorsen_simple_cm(k::Real, heave_amp::Number, pitch_amp::Number; 
    pitch_loc::Real=0, reference_loc::Real=0, chord::Real=1, U::Real=1)
    @assert(k > 0, "Chord reduced frequency must be positive.")
    @assert(pitch_loc == pitch_loc, "Pitch location must be a normal number.")
    @assert(chord > 0, "Chord must be positive")
    @assert(U > 0)
    ck = theodorsen_fn(k)
    h0s = heave_amp / chord;
    alpha0 = pitch_amp
    xp2 = pitch_loc / 2 + 0.5
    xm2 = reference_loc / 2 + 0.5

    cmh = 2 * pi * h0s * (-2 * im * k * ck * (xm2 - 1/4) + k^2 * (xm2 - 1/2))
    cma = 2 * pi * alpha0 * (
        ck * (1 - 2 * im * k * (xp2 - 3/4)) * (xm2 - 1/4)
        + k^2 * (xp2 * (xm2 - 1/2) - 1 / 2 * (xm2 - 9/16))
        + im * k / 2 * (xm2 - 3/4))

    return cmh + cma
end

"""
Evaluate a simple Theodorsen like problem.
pitch_loc maps -1 to LE, 1 to TE.
"""
function theodorsen_simple_bound_vorticity(k::Real, heave_amp::Number, pitch_amp::Number; 
    pitch_loc::Real=0, reference_loc::Real=0, chord::Real=1, U::Real=1)
    @assert(k > 0, "Chord reduced frequency must be positive.")
    @assert(pitch_loc == pitch_loc, "Pitch location must be a normal number.")
    @assert(chord > 0, "Chord must be positive")
    @assert(U > 0)
    ck = theodorsen_fn(k)
    h0s = heave_amp / chord;
    alpha0 = pitch_amp
    xp2 = pitch_loc / 2 + 0.5
    xm2 = reference_loc / 2 + 0.5
    h20 = SpecialFunctions.hankelh2(0, k)
    h21 = SpecialFunctions.hankelh2(1, k)

    tcommon = 4 * U * chord * exp(-im * k) / (im * h20 + h21)
    bvh = tcommon * h0s
    bva = tcommon * alpha0 * ((xp2 - 3/4) - 1 / (2 * im * k))
    return bvh + bva
end

"""
Evaluate a simple Theodorsen like problem.
pitch_loc maps -1 to LE, 1 to TE.
"""
function theodorsen_simple_a0(k::Real, heave_amp::Number, pitch_amp::Number; 
    pitch_loc::Real=0, chord::Real=1, U::Real=1)
    @assert(k > 0, "Chord reduced frequency must be positive.")
    @assert(pitch_loc == pitch_loc, "Pitch location must be a normal number.")
    @assert(chord > 0, "Chord must be positive")
    @assert(U > 0)
    ck = theodorsen_fn(k)
    h0s = heave_amp / chord;
    alpha0 = pitch_amp
    xp2 = pitch_loc / 2 + 0.5
    omega = 2 * U * k / chord
    alphadot = im * omega * pitch_amp
    h20 = SpecialFunctions.hankelh2(0, k)
    h21 = SpecialFunctions.hankelh2(1, k)

    w3qch = - 2 * im * U * k * h0s
    w3qca = U * (1 - 2 * im * k * (xp2 - 3/4)) * alpha0
    a0 = ck * (w3qch + w3qca) / U - alphadot * chord / (4 * U)
    return a0
end

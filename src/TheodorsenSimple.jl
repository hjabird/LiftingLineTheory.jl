
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
    fq = 2 * U * k / chord

    t11 = pi * chord/2
    t121 = fq^2 * heave_amp / U^2
    t122 = im * fq * pitch_amp / U
    t123 = (chord/2) * pitch_loc * fq^2 * pitch_amp / U^2
    t12 = t121 + t122 + t123
    t1 = t11 * t12

    t21 = 2 * pi * theodorsen_fn(k)
    t221 = -im * fq * heave_amp/ U
    t222 = pitch_amp
    t223 = (chord/2) * (1/2 - pitch_loc) * im * fq * pitch_amp / U
    t22 = t221 +  t222 + t223
    t2 = t21 * t22
    return t1 + t2
end

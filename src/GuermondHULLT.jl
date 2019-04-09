

mutable struct GuermondULLT
    angular_fq :: Real              # in rad / s
    free_stream_vel :: Real

    wing :: CurvedAnalyticWing
    amplitude_fn :: Function        # Amplitude of oscillation wrt/ span pos.
    pitch_plunge :: Int64           # Plunge = 3, Pitch = 5. Otherwise invalid.


end


function G_fn(a::GuermondULLT, eta :: Real)
    # G = exp(im * k * c_t(eta)) * bound_vort
end

function H_fn(a::GuermondULLT, x :: Real, y :: Real)
    U = a.free_stream_vel
    c = a.wing.chord_fn(y)
    omega = a.angular_fq
    k = omega * c / (2 * U)
    @assert(k < 1, "Current H formulation only valid for small k!")
    


end
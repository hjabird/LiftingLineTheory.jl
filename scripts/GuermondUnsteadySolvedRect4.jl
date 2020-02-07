
@everywhere using LiftingLineTheory
using PyPlot

let 
    @everywhere wing = make_rectangular(StraightAnalyticWing, 4, 4)
    @everywhere dwsinusoidal = true

    srfs = collect(0.01:0.2:6)
    omegas = srfs ./ wing.semispan

    @everywhere function getcl3d(omega)
        local prob = GuermondUnsteady2(omega, wing; sinusoidal_downwash=dwsinusoidal)
        compute_collocation_points!(prob)
        compute_fourier_terms!(prob)
        local cl = lift_coefficient(prob)
        return cl / (im * omega)
    end
    @everywhere function getcl2d(omega)
        local prob = GuermondUnsteady2(omega, wing; sinusoidal_downwash=dwsinusoidal)
        local cl = lift_coefficient(prob; order=0)
        return cl / (im * omega)
    end

    cl2ds = pmap(getcl2d, omegas)
    cl3ds = pmap(getcl3d, omegas)
    plot(srfs, abs.(cl2ds))
    plot(srfs, abs.(cl3ds))
end
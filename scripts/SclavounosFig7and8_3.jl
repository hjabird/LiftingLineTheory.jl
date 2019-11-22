
using LiftingLineTheory
using PyPlot

let
    println("Note that due to a difference between this implementation"*
        " and Sclavounos' implementation, the high frequency response "*
        " differs. See Bird & Ramesh, 2019, TCFD.")
    span = 1
    aspect_ratio = 4
    wing = LiftingLineTheory.make_rectangular(StraightAnalyticWing, 
        aspect_ratio, span)
    srfs = collect(0.01:0.2:8)
    omegas = 2 * srfs / span;
    cl_abs = zeros(length(srfs))
    cl_ph = zeros(length(srfs))
    stcl_abs = zeros(length(srfs))
    stcl_ph = zeros(length(srfs))
    swfcl_abs = zeros(length(srfs))
    swfcl_ph = zeros(length(srfs))
    for i = 1 : length(srfs)
        prob = HarmonicULLT3(omegas[i], wing; num_terms = 9)
        compute_collocation_points!(prob)
        compute_fourier_terms!(prob)
        cl = lift_coefficient(prob)
        cl_abs[i] = abs(cl)
        cl_ph[i] = atan(imag(cl), real(cl))

        prob = HarmonicULLT3(omegas[i], wing; downwash_model=strip_theory)
        compute_collocation_points!(prob)
        compute_fourier_terms!(prob)
        stcl = lift_coefficient(prob)
        stcl_abs[i] = abs(stcl)
        stcl_ph[i] = atan(imag(stcl), real(stcl))

        prob = HarmonicULLT3(omegas[i], wing; downwash_model=streamwise_filaments)
        compute_collocation_points!(prob)
        compute_fourier_terms!(prob)
        swfcl = lift_coefficient(prob)
        swfcl_abs[i] = abs(swfcl)
        swfcl_ph[i] = atan(imag(swfcl), real(swfcl))
    end

    figure()
    ax = gca()
    ax.imshow(imread("scripts/Sclav87ref/Fig7.PNG"), extent=[0, 8, 0, 8])
    ax.set_aspect("auto")
    plot(srfs, cl_abs, label="ULLT")
    plot(srfs, stcl_abs, label="Strip theory")
    plot(srfs, swfcl_abs, label="Streamwise filemants")
    xlabel("Span reduced frequency")
    ylabel("ABS(C_L)")
    legend()

    figure()
    ax = gca()
    ax.imshow(imread("scripts/Sclav87ref/Fig8.PNG"), extent=[0, 8, -270, 0])
    ax.set_aspect("auto")
    plot(srfs, ((rad2deg.(cl_ph) .+ 360) .% 360) .-360, label="ULLT")
    plot(srfs, ((rad2deg.(stcl_ph) .+ 360) .% 360) .-360, label="Strip theory")
    plot(srfs, ((rad2deg.(swfcl_ph) .+ 360) .% 360) .-360, label="Streamwise filemants")
    xlabel("Span reduced frequency")
    ylabel("Ph(C_L)")
    legend()
    return srfs, cl_abs, cl_ph
end
#=
Compare LAUTAT to theodorsen in heave for a single kinematic.
=#

using LiftingLineTheory
using CVortex
using PyPlot

let
    k = 0.4
    dtstar = 0.0075
    amp_heave = 0.01
    amp_pitch = deg2rad(0.0)
    oscl = 6
    omega = 2 * k
    T = 2 * pi / omega
    tmax = oscl * T
    println("k=",k)
    println("H_amp=",amp_heave)
    println("oscl=",oscl)
    println("TMAX=", tmax)


    hvfn = t->amp_heave * cos(omega * t)
    ptfn = t->amp_pitch * cos(omega * t)
    prob = LAUTAT(;
        kinematics=RigidKinematics2D(hvfn, ptfn, 0.0),
        dt=dtstar,
        regularisation=singular_regularisation())
    hdr = csv_titles(prob)
    rows = zeros(0, length(hdr))
    nsteps = Int64(ceil(tmax/dtstar))
    println("nsteps=", nsteps)
    for i = 1 : nsteps
        advance_one_step(prob)
        rows = vcat(rows, csv_row(prob))
    end


    figure()
    theod_amp = LiftingLineTheory.theodorsen_simple_cl(k, amp_heave, amp_pitch)
    xs = collect(0:dtstar:tmax)
    yst = real.(theod_amp * exp.(im * omega .* xs))
    plot(xs, yst, label="Theodorsen")
    plot(rows[50:end, 1], rows[50:end, 9], label="LAUTAT")
    xlabel("Time")
    ylabel("C_L normalised with theodorsen CL")
    legend()
    
    println("ABS_Theod=", abs(theod_amp))
    println("Ph_Theod", atan(imag(theod_amp), real(theod_amp)))

    # Get last wave indices...
    oclm = Int64(ceil((oscl-1)*T/dtstar))
    cllast = rows[oclm:end, 9]
    clmax = maximum(cllast)
    clmin = minimum(cllast)
    println("ABS_LAUTAT=", (clmax-clmin)/2)
    println("REL_ABS_ERR=", ((clmax-clmin)/2-abs(theod_amp))/abs(theod_amp))

    return 
end

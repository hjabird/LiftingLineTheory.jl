#=
Compare LAULLT to CFD for Eldredge case with parameters:
    Wing:
        Rectangular
        AR=3
    Kinematics:
        Eldredge quarter chord pitch
        amplitude = 25 degrees
        T1 = 1
        T2 = 3
        T3 = 4
        T4 = 6
        chord = 1
        U_inf = 1
        a = 11
=#

module_path = dirname(pathof(LiftingLineTheory))*"/../"
include(module_path * "scripts/LAULLT/RunCFDComparison.jl")
cfd_path = module_path * "scripts/CFD_results_ref/" 

let
    AR = 3
    wing = LiftingLineTheory.make_rectangular(StraightAnalyticWing, AR, AR)

    t0 = 0
    t1 = 1
    t2 = 3
    t3 = 4
    t4 = 6
    tmax = 7
    amp = deg2rad(25)
    a=11
    U_inf = 1.
    chord = 1
	
	println("Comparing LAULLT CFD in pitch in "*Base.source_path())

    wing = make_rectangular(StraightAnalyticWing, 4, 4)
    eld_fn = t->eldredge_ramp(t, t1, t2, t3, t4, U_inf, chord; a=a)
    eld_max = eld_fn((t2+t3)/2)
    kinematics = make_pitch_function(RigidKinematics2D, -0.25, t->amp * eld_fn(t) / eld_max)
    
    run_laullt_cfd_comp(
        wing, kinematics,
        cfd_path*"eldredge_pitchQC25_Re10k_1_3_4_6_RectAr3.dat";
        cfd_res_path2d=cfd_path*"eldredge_pitchQC25_Re10k_1_3_4_6_RectArInf.dat",
        cfd_res_toff=250
    )
end

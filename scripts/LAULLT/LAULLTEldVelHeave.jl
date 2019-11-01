#
# LAULLTEldVelHeave.jl
#
# Use the Eldredge function to control the heave velocity.
#
################################################################################

using PyPlot
using QuadGK

let
    # Kinematic parameters:
    t0 = 0
    t1 = 1
    t2 = 3
    t3 = 4
    t4 = 6
    tmax = 7
    a_val = 11
    U_inf = 1
    nominal_chord = 1
    max_vel = tan(deg2rad(3)) # Equivalent AoA. 3 degree ~ 0.05m/s

    # Wing parameters
    aspect_ratio = 4
    span = 4
    wing = make_rectangular(StraightAnalyticWing, 4, 4)
    num_divisions = 32 # Number of strips in LAULLT
    lattice_refinement=3

    # The main challenge is creating the kinematics.
    println("Generating kinematics...")
    amp = max_vel * 1 / eldredge_ramp(
            (t3+t2)/2, t1, t2, t3, t4, a_val, U_inf, nominal_chord)
    function heave_vel(t::Real)
        return amp * eldredge_ramp(t, t1, t2, t3, t4, a_val, U_inf, nominal_chord)
    end
    function heave_disp(t::Real)
        val, err = quadgk(heave_vel, 0, t)
        return val
    end
    kinem = LiftingLineTheory.make_plunge_function(RigidKinematics2D, heave_disp, heave_vel)
    # But the whole integration thing is horribly slow, so lets create an interpolated version.
    interpolation_dt = 0.025
    int_mat = csv_row(kinem, collect(t0:interpolation_dt:tmax))
    kinem = from_matrix(RigidKinematics2D, int_mat) # w/ cubic spline interpolation

    #position_remap = y->sin(y * pi/2)
    position_remap = y->y
    prob = LAULLT(wing, kinem; 
        num_inner_solutions=num_divisions,
        outer_lattice_refinement=lattice_refinement,
        inner_solution_mapping=position_remap,
        outer_lattice_mapping=position_remap)

    # Time marching...
    nsteps = Int64(ceil((tmax-t0)/prob.inner_sols[1].dt))
    hdr = csv_titles(prob)
    rows = zeros(0, length(hdr))
    for i = 1 : nsteps
        print("\rStep ", i, " of ", nsteps, ".\t\t\t\t\t")
        advance_one_step(prob)
        to_vtk(prob, "test_"*string(i); translation=[0,0,prob.inner_sols[1].kinematics.z_pos(prob.inner_sols[1].current_time)])
        rows = vcat(rows, csv_row(prob))
        for j = 1 : num_divisions
            yp = prob.inner_sol_positions[j] * wing.semispan
            to_vtk(prob.inner_sols[j], "inner"*string(j)*"_"*string(i);
                streamdir=[1,0,0], updir=[0,0,1], 
                translation=[-wing.chord_fn(yp)/2, yp, 0])
        end
    end

    println(hdr)
    plot(rows[:,1], rows[:,5], label="8innerInterp4")
end #let

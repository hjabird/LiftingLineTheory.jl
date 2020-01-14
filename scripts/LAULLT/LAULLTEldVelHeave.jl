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
    t1 = 2.5
    t2 = 7.5
    t3 = 10
    t4 = 15
    tmax = 17.5
    sigma_val = 0.5#0.8878
    U_inf = 1
    nominal_chord = 1
    max_vel = tan(deg2rad(3)) # Equivalent AoA. 3 degree ~ 0.05m/s

    # Wing parameters
    aspect_ratio = 3
    span = 3
    wing = make_rectangular(StraightAnalyticWing, 4, 4)
    num_divisions = 8 # Number of strips in LAULLT
    lattice_refinement= 1

    # The main challenge is creating the kinematics.
    println("Generating kinematics...")
    amp = max_vel * 1 / eldredge_ramp(
            (t3+t2)/2, t1, t2, t3, t4, U_inf, nominal_chord; sigma=sigma_val) 
    function heave_vel(t::Real)
        return amp * eldredge_ramp(t, t1, t2, t3, t4, U_inf, nominal_chord; sigma=sigma_val)
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

    println("Kinematics:")
    println("\tt1=",t1)
    println("\tt2=",t2)
    println("\tt3=",t3)
    println("\tt4=",t4)
    println("\tsimga=",sigma_val)
    println("\tmax_vel=",max_vel)
    println("\tmax displacement=", maximum(int_mat[:,2]))

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
    #=
    for i = 1 : nsteps
        print("\rStep ", i, " of ", nsteps, ".\t\t\t\t\t")
        advance_one_step(prob)
        #to_vtk(prob, "test_"*string(i); translation=[0,0,prob.inner_sols[1].kinematics.z_pos(prob.inner_sols[1].current_time)])
        rows = vcat(rows, csv_row(prob))
        for j = 1 : num_divisions
            yp = prob.inner_sol_positions[j] * wing.semispan
            #to_vtk(prob.inner_sols[j], "inner"*string(j)*"_"*string(i);
            #    streamdir=[1,0,0], updir=[0,0,1], 
            #    translation=[-wing.chord_fn(yp)/2, yp, 0])
        end
    end

    println(hdr)
    figure()
    plot(rows[:,1], rows[:,5], label="CL")=#
    figure()
    plot(int_mat[:,1], int_mat[:,2], label="Kinem")
    xlabel("t*")
    ylabel("Displacement")
    figure()
    plot(int_mat[:,1], int_mat[:,3], label="Vel")
    xlabel("t*")
    ylabel("Velocity")
end #let

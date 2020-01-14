#=
Use the UVLM method to simulate a vortex sheet rolling up.
=#

let
    ni = 40
    nj = 40
    geomfunc = (x,y)->[x,y,0]
    dipole_str = (x,y)->1
    vl = VortexLattice(geomfunc, dipole_str; 
        xs=collect(range(-1, stop=1, length=ni)), ys=collect(range(-1, stop=1, length=nj)))
    
    # Now we can convect...
    steps = 100
    for i = 1 : steps
        points = get_vertices(vl)
        fil_start, fil_ends, fil_str = to_filaments(vl)
        to_vtk(VortexFilament, fil_start, fil_ends, fil_str, "test_"*string(i))
        vels = filament_induced_velocity(fil_start, fil_ends, fil_str, points)
        points += dt * vels
        set_vertices!(vl, points)
    end

end

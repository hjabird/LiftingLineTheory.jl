#
# PointsVortsODE.jl
#
# Methods for solving ODEs involving a fixed number of particles and filaments.
# Or rather, with a points field and vorticity of possibly different dimensions,
# but most importantly of the same size between substeps. 
#
# Copyright HJAB 2020
#
################################################################################

mutable struct PointsVortsODE
    # y_n funcs.
    get_dpoints :: Function    # func(void) :: Matrix{Float}
    get_dvorts :: Function # func(void) :: Matrix{Float}

    # dy/dt funcs
    vel_method :: Function # func(void) :: Matrix{Float}
    dvort_method :: Function # func(void) :: Matrix{Float}
    
    # set y_n funcs.
    set_dpoints :: Function # func() :: Nothing
    set_dvorts :: Function # func() :: Nothing

    function PointsVortsODE(
        get_dpoints :: Function,
        get_dvorts :: Function,
        vel_method :: Function,
        dvort_method :: Function,
        set_dpoints :: Function, 
        set_dvorts :: Function )

        @assert(hasmethod(get_dpoints, Tuple{}))
        @assert(hasmethod(get_dvorts, Tuple{}))
        @assert(hasmethod(vel_method, Tuple{}))
        @assert(hasmethod(dvort_method, Tuple{}))
        @assert(hasmethod(set_dpoints, Tuple{Matrix{Float32}}))
        @assert(hasmethod(set_dvorts, Tuple{Matrix{Float32}}))

        new(get_dpoints, get_dvorts,
            vel_method, dvort_method,
            set_dpoints, set_dvorts)
    end
end

function euler_step(a::PointsVortsODE, dt) :: Nothing
    # Simple Euler method
    # y_n+1 = un + h f'(xn, yn) + O(h^2)
    points = a.get_dpoints()
    vorts = a.get_dvorts()
    vels = a.vel_method()
    dvorts = a.dvort_method()
    points += dt * vels
    vorts += dt * dvorts
    a.set_dpoints(points)
    a.set_dvorts(vorts)
    return
end

function runge_kutta_2_step(a::PointsVortsODE, dt) :: Nothing
    # Runge Kutta 2nd order method:
    #   k1 = h f'(xn, yn)
    #   k2 = h f'(xn + h/2, yn + k1/2)
    #   y_n+1 = yn + k2 + O(h^3)
    points_yn = a.get_dpoints()
    vorts_yn = a.get_dvorts()

    # k1 
    vels = a.vel_method()
    dvorts = a.dvort_method()
    points_yph = points_yn + (dt / 2) * vels
    vorts_yph = vorts_yn + (dt / 2) * dvorts

    # k2
    a.set_dpoints(points_yph)
    a.set_dvorts(vorts_yph)
    vels = a.vel_method()
    dvorts = a.dvort_method()
    points = points_yn + dt * vels
    vorts = vorts_yn + dt * dvorts

    a.set_dpoints(points)
    a.set_dvorts(vorts)
    return
end

function runge_kutta_4_step(a::PointsVortsODE, dt) :: Nothing
    # Runge Kutta 4th order method:
    #   k1 = h f'(xn, yn)
    #   k2 = h f'(xn + h/2, yn + k1/2)
    #   k3 = h f'(xn + h/2, yn + k2/2)
    #   k4 = h f'(xn + h, yn + k3)
    #   y_n+1 = yn + k1/6 + k2/3 + k3/3 +k4/6 + O(h^5)
    points_yn = a.get_dpoints()
    vorts_yn = a.get_dvorts()
    pts_sz = size(points_yn)
    vorts_sz = size(vorts_yn)

    # k1 
    vels_k1 = a.vel_method()
    dvorts_k1 = a.dvort_method()
    @assert(pts_sz == size(vels_k1), 
        "Number of velocities different to num points.")
    @assert(vorts_sz == size(dvorts_k1), 
        "Number of vorticities different to num particles.")
    # k2
    a.set_dpoints(points_yn + (dt / 2) * vels_k1)
    a.set_dvorts(vorts_yn + (dt / 2) * dvorts_k1)
    vels_k2 = a.vel_method()
    dvorts_k2 = a.dvort_method()
    @assert(pts_sz == size(vels_k2), 
        "Number of velocities different to num points.")
    @assert(vorts_sz == size(dvorts_k2), 
        "Number of vorticities different to num particles.")
    # k3
    a.set_dpoints(points_yn + (dt / 2) * vels_k2)
    a.set_dvorts(vorts_yn + (dt / 2) * dvorts_k2)
    vels_k3 = a.vel_method()
    dvorts_k3 = a.dvort_method()
    @assert(pts_sz == size(vels_k3), 
        "Number of velocities different to num points.")
    @assert(vorts_sz == size(dvorts_k3), 
        "Number of vorticities different to num particles.")
    # k4
    a.set_dpoints(points_yn + dt * vels_k3)
    a.set_dvorts(vorts_yn + dt * dvorts_k3)
    vels_k4 = a.vel_method()
    dvorts_k4 = a.dvort_method()
    @assert(pts_sz == size(vels_k4), 
        "Number of velocities different to num points.")
    @assert(vorts_sz == size(dvorts_k4), 
        "Number of vorticities different to num particles.")

    points = points_yn + dt * (
        vels_k1 ./ 6 + vels_k2 ./ 3 +
        vels_k3 ./ 3 + vels_k4 ./ 6 )
    vorts = vorts_yn + dt * (
        dvorts_k1 ./ 6 + dvorts_k2 ./ 3 +
        dvorts_k3 ./ 3 + dvorts_k4 ./ 6 )
    a.set_dpoints(points)
    a.set_dvorts(vorts)
    return
end

function backwards_euler(a::PointsVortsODE, dt) :: Nothing
    # Backwards euler method - see wikipedia
    # y_n+1 = un + h f'(xn, yn) + O(h^2)
    points = a.get_dpoints()
    vorts = a.get_dvorts()
    # forwards euler step first
    vels = a.vel_method()
    dvorts = a.dvort_method()
    points_yp = points + dt * vels
    vorts_yp = vorts + dt * dvorts
    a.set_dpoints(points_yp)
    a.set_dvorts(vorts_yp)
    # And iterate
    for i = 1 : 16
        vels = a.vel_method()
        dvorts = a.dvort_method()
        points_yp = points + dt * vels
        vorts_yp = vorts + dt * dvorts
        a.set_dpoints(points_yp)
        a.set_dvorts(vorts_yp)
    end
    return
end


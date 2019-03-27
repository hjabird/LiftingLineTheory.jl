#
# HarmonicDownwash.jl
#
# Copyright HJA Bird 2019
#
#============================================================================#

@enum HarmonicDownwash begin
	heaving     # Heaving x->1
    pitching    # Pitching x->x
end

function downwash_fn(a::HarmonicDownwash)
    if( a == heaving )
        function fn(x::Real)
            return 1
        end
    elif( a == pitching )
        function fn(x::Real)
            return x
        end
    else
        error("Not defined for this harmonic downwash enum")
    end
    return fn
end

function pressure_distribution_fn(a::HarmonicDownwash)
    if( a == heaving )
        fn = harmonic_pressure_distribution_uniform_downwash
    elif( a == pitching )
        fn = harmonic_pressure_distribution_pitching_downwash
    else
        error("Not defined for this harmonic downwash enum")
    end
    return fn
end

function bound_vorticity_fn(a::HarmonicDownwash)
    if( a == heaving )
        fn = harmonic_bound_vorticity_uniform_downwash
    elif( a == pitching )
        fn = harmonic_bound_vorticity_pitching_downwash
    else
        error("Not defined for this harmonic downwash enum")
    end
    return fn
end

function lift_coefficient_fn(a::HarmonicDownwash)
    if( a == heaving )
        fn = harmonic_lift_coeff_uniform_downwash
    elif( a == pitching )
        fn = harmonic_lift_coeff_pitching_downwash
    else
        error("Not defined for this harmonic downwash enum")
    end
    return fn
end

function harmonic_pressure_distribution_uniform_downwash(
    x :: Real, k :: Real)

    # See HJAB Notes 6 pg 2
    @assert(abs(x) <= 1/2, "x must be on chord defined as [-1/2, 1/2]")
    xl = x + 1/2;
    
    t1 = 2*(2*(1-theodorsen_fn(k)) + 1) * sqrt((1 - xl) / xl)
    coeff = 8 * im * k
    t2 = coeff * xl
    t3 = -coeff * xl^2
    t4 = coeff * xl * log(4 * xl - 1) / pi
    result = t1 + t2 + t3 + t4    
    return result
end

function harmonic_pressure_distribution_pitching_downwash(
    x :: Real, k :: Real)

    # See HJAB Notes 6 pg 3
    @assert(abs(x) <= 1/2, "x must be on chord defined as [-1/2, 1/2]")
    xl = x + 1/2;
    
    t1 = (3/4)*(2*(1-theodorsen_fn(k)) + 1) * sqrt((1 - xl) / xl)
    t2 = -4 * k * im * xl^3
    t3 = 2 * k * im * xl^2
    t4 = 2 * k * im * xl
    t5 = 4 * k * im * xl^2 * ln(4 * xl - 1) / pi
    result = t1 + t2 + t3 + t4    
    # In notes, x is [0, c] and fn is x->x (pitch about LE.) Correction:
    result -= 0.5 * harmonic_pressure_distribution_uniform_downwash(x, k)
    return result
end

function harmonic_bound_vorticity_uniform_downwash(
    semichord :: Real, k :: Real, U :: Real)

    num = 4 * U * exp(-im * k)
    den = im * SpecialFunctions.hankelh2(0, norm_fq * semichord) +
        SpecialFunctions.hankelh2(1, norm_fq * semichord)
    return num / den
end

function harmonic_bound_vorticity_pitching_downwash(
    semichord :: Real, k :: Real, U :: Real)

    num = -4 * U * exp(-im * k) * (semichord/(k*im) + semichord / 2)
    den = im * SpecialFunctions.hankelh2(0, k) +
        SpecialFunctions.hankelh2(1, k)
    return num / den
end

function harmonic_lift_coeff_uniform_downwash(
    semichord :: Real, k :: Real, U :: Real)

    # HJAB Notes 5 pg 53
    t1 = -2 * pi * theodorsen_fn(k) 
    t2 = -im * pi * k 
    return t1 + t2
end


function harmonic_lift_coeff_pitching_downwash(
    semichord :: Real, k :: Real, U :: Real)

    # Notes 5 pg 53
    t1 = pi * semichord
    t21 = 2 * theodorsen_fn(k) * pi * (semichord * k)
    t22 = 1 + im * k / 2
    t2 = t21 * t22
    return t1 + t2
end

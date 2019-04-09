#
# HarmonicDownwash.jl
#
# Copyright HJA Bird 2019
#
#============================================================================#

function harmonic_acc_pot_diff(a::LinearExpression, omega :: Real, chord :: Real, U :: Real)
    ret = LinearExpression()
    for term in a.terms
        if( term.type == LinExprTerm_const )
            fn = harmonic_acc_pot_diff_LinExprTerm_const(term, omega, chord, U)
        elif( term.type == LinExprTerm_x )
            fn = harmonic_acc_pot_diff_LinExprTerm_x(term, omega, chord, U)
        else
            error("Not defined linear term")
        end
        ret += fn
    end
    simplify!(ret)
    return ret
end

function harmonic_acc_pot_diff_LinExprTerm_const(
    term :: LinearExpressionTerm, omega::Real, chord::Real, U::Real)

    cp_to_psi = U^2 / 2
    k = omega * chord / (2 * U)
    ret = LinearExpression()

    # HJAB Notes 6 pg 2
    t = 2*(2*(1-theodorsen_fn(k))+chord)
    push!(ret.terms, LinearExpressionTerm(LinExprTerm_sqrt_cmxox, t))
    t = 4 * im * omega * chord / U 
    push!(ret.terms, LinearExpressionTerm(LinExprTerm_x, t))
    t = - 4 * im * omega  / U
    push!(ret.terms, LinearExpressionTerm(LinExprTerm_x2, t))
    t = 4 * im * omega / (U * pi)
    push!(ret.terms, LinearExpressionTerm(LinExprTerm_x_ln4xmcoc, t))

    ret *= cp_to_psi * term.coeff
    return result
end

function harmonic_acc_pot_diff_LinExprTerm_x(
    term :: LinearExpressionTerm, omega::Real, chord::Real, U::Real)

    cp_to_psi = U^2 / 2
    k = omega * chord / (2 * U)
    ret = LinearExpression()

    # HJAB Notes 6 pg 2
    t = 3 * chord * (2*(1-theodorsen_fn(k))+chord) / 4
    push!(ret.terms, LinearExpressionTerm(LinExprTerm_sqrt_cmxox, t))
    t = im * omega * chord^2 / U
    push!(ret.terms, LinearExpressionTerm(LinExprTerm_x, t))
    t = im * omega * c / U
    push!(ret.terms, LinearExpressionTerm(LinExprTerm_x2, t))
    t = -2 * im * omega / U
    push!(ret.terms, LinearExpressionTerm(LinExprTerm_x3, t))
    t = 2 * im * omega / (U * pi)
    push!(ret.terms, LinearExpressionTerm(LinExprTerm_x2_ln4xmcoc, t))

    ret *= cp_to_psi * term.coeff
    return result
end

function harmonic_bound_vort(a::LinearExpression, omega :: Real, chord :: Real, U :: Real)
    ret = LinearExpression()
    for term in a.terms
        if( term.type == LinExprTerm_const )
            fn = harmonic_acc_pot_diff_LinExprTerm_const(term, omega, chord, U)
        elif( term.type == LinExprTerm_x )
            fn = harmonic_acc_pot_diff_LinExprTerm_x(term, omega, chord, U)
        else
            error("Not defined linear term")
        end
        ret += fn
    end
    simplify!(ret)
    return ret
end

function harmonic_bound_vort_LinExprTerm_const(
    a::LinearExpressionTerm, omega :: Real, chord :: Real, U :: Real)

    @assert(term.type == LinExprTerm_const)

    semichord = chord / 2
    k = omega * chord / (2 * U)
    norm_fq = omega / U
    num = 4 * U * exp(-im * k)
    den = im * SpecialFunctions.hankelh2(0, norm_fq * semichord) +
        SpecialFunctions.hankelh2(1, norm_fq * semichord)
    return a.coefficient * num / den
end

function harmonic_bound_vort_LinExprTerm_x(
    a::LinearExpressionTerm, omega :: Real, chord :: Real, U :: Real)

    @assert(term.type == LinExprTerm_x)

    semichord = chord / 2
    k = omega * chord / (2 * U)
    norm_fq = omega / U
    num = -4 * U * exp(-im * k) * (semichord/(k*im) + semichord / 2)
    den = im * SpecialFunctions.hankelh2(0, k) +
        SpecialFunctions.hankelh2(1, k)
    return a.coefficient * num / den
end

#=
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
=#

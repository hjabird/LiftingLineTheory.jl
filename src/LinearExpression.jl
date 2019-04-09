#
# LinearExpression.jl
#
# Copyright HJA Bird 2019
#
#============================================================================#

@enum LinearExpressionTermType begin
    # See evaluate(::LinearExpressionTerm, x, c) for how evaluated.
    LinExprTerm_const           # Constant value of 1
    LinExprTerm_x               # x
    LinExprTerm_x2              # x^2
    LinExprTerm_x3              # x^3
    LinExprTerm_sqrt_cmxox      # sqrt((c - x) / x) where c is chord
    LinExprTerm_x_ln_4xmcoc     # x * ln((4*x - c) / c)
    LinExprTerm_x2_ln_4xmcoc    # x^2 * ln((4*x - c) / c)
end

mutable struct LinearExpressionTerm
    type :: LinearExpressionTermType
    coefficient :: Real
    function LinearExpressionTerm(t::LinearExpressionTermType, coeff::Real)
        new(t, coeff)
    end
end

mutable struct LinearExpression
    terms :: Vector{LinearExpressionTerm}
    function LinearExpression()
        new(Vector{LinearExpressionTerm}(undef, 0))
    end
    function LinearExpression(a::Vector{LinearExpressionTerm})
        new(a)
    end
end

function simplify!(a::LinearExpression)
    i = 1
    while (i <= length(a.terms))
        type = a.terms[i].type
        if a.terms[i].coefficient == 0
            a = deleteat!(a, i)
            continue
        end
        j = i + 1
        while j  <= length(a.terms)
            if a.terms[j].type == type
                ci = a.terms[i].coefficient
                cj = a.terms[j].coefficient
                if cj == 0
                    continue
                end
                a.terms[i].coefficient = ci + cj
            end
        end
        if a.terms[i].coefficient == 0
            a = deleteat!(a, i)
            continue
        end
        i += 1
    end
    return a
end

function evaluate(a::LinearExpression, x::Real, chord::Real)
    value = 0.0
    for i = 1 : length(a.terms)
        
    end
end

function evaluate(a::LinearExpressionTermType, x::Real, chord::Real)
    if a == LinExprTerm_const
        ret = 1
    elseif a == LinExprTerm_x
        ret = x 
    elseif a == LinExprTerm_x2
        ret = x^2
    elseif a == LinExprTerm_x3
        ret = x^3
    elseif a == LinExprTerm_sqrt_cmxox       # sqrt((c - x) / x) where c is chord
        ret = sqrt((chord - x) / x)
    elseif a == LinExprTerm_x_ln_4xmcoc      # x * ln((4*x - c) / c)
        ret = x * log((4 * x - chord) / chord)    
    elseif a == LinExprTerm_x2_ln_4xmcoc    # x^2 * ln((4*x - c) / c)
        ret = x^2 * log((4 * x - chord) / chord)    
    else
        error("Undefined LineExpressionTermType.")
    end
    return ret
end

import Base
function Base.:+(a::LinearExpression, b::LinearExpression)
    ret = a
    for i = 1 : length(b.terms)
        push!(a.terms, b.terms[i])
    end
    simplify!(a)
    return ret
end

function Base.:*(a::LinearExpression, b::Real)
    c = deepcopy(a)
    for i = 1 : length(a.terms)
        c.terms[i].coefficient *= b 
    end
    return c
end

#
# shape.jl
#
# Copyright HJA Bird 2019
#
#	Plot the shape of a generated wing.
#
#============================================================================#

push!(LOAD_PATH, "../src")
using LiftingLineTheory
using PyPlot

semispan = 1
AR = 8
wing = LiftingLineTheory.make_van_dyke_cusped(StraightAnalyticWing, AR, semispan * 2, 4)
srf = 1
wa = area(wing)

println("Semispan = ", semispan, ", AR = ", AR, ", area = ", wa)
println("Computed AR = ", aspect_ratio(wing))
y = collect( -wing.semispan: 0.05: wing.semispan)
c = map(x->wing.chord_fn(x), y)

figure()
plot(y, c)
title("Wing chord distribution")
xlabel(L"y")
ylabel(L"chord")
axis("equal")

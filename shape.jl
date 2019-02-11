include("sclavounos.jl")
using PyPlot


semispan = 1
aspect_ratio = 8
wing = make_van_dyke_cusped(StraightAnalyticWing, aspect_ratio, semispan * 2, 4)
srf = 1
area = wing_area(wing)

println("Semispan = ", semispan, ", AR = ", aspect_ratio, ", area = ", area)
println("Computed AR = ", calculate_aspect_ratio(wing))
y = collect( -wing.semispan: 0.05: wing.semispan)
c = map(x->wing.chord_fn(x), y)

figure()
plot(y, c)
title("Wing chord distribution")
xlabel(L"y")
ylabel(L"chord")
axis("equal")

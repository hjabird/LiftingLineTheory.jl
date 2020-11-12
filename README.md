# LiftingLineTheory.jl

A collection of aerodynamics research codes for unsteady flow written in Julia.
The interface to this library is constantly changing. The code may contain bugs.

## Codes:

2D:

* LAUTAT: Large amplitude unsteady lifting-line theory
* LDVM: Leading edge suction parameter modulated discrete vortex method
* VortexLatticeMethod2D: Like LDVM, except with a velocity boundary condition solved using a linear system.
* Theodorsen like methods.

3D:

* Oscillatory unsteady lifting-line theory for small amplitude problems (Sclavounos & variants. Non-working Guermond & Sellier like code).
* LAULLT: Large amplitude unsteady lifting-line theory. Unsteady lifting-line theory based upon LAUTAT for inner solutions.
* Vortex lattice method. 
* PEVLM: Particle enhanced vortex lattice method. The VLM, except the wake is converted into regularised vortex particles.
* LMPEVLM / VoFFLE: LESP modulated particle enhanced vortex lattice method / VOrtex Formation on Finite Leading Edge model. PEVLM like model, except it also have leading edge vortices.

Also supporting functions.

Copyright HJA Bird.



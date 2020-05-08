#
# VortexLattice.jl
#
# A square vortex lattice grid.
#
# Copyright HJA Bird 2019 - 2020
#
#==============================================================================#

mutable struct VortexLattice
    vertices :: Array{Float64, 3}   # Arranged [i, j, k] where i is row, 
                                    # j is col and k is index of [x,y,z].
    strengths :: Matrix{Float64}    # Arranged [i, j] where i is row, j is col.

    function VortexLattice(
        vertices::Array{Float64, 3}, strengths::Matrix{Float64})
        vl = new(vertices, strengths)
        @assert(check(vl))
        return vl
    end

    function VortexLattice(vertices::Array{Float64, 3})
        @assert(size(vertices)[3]==3, "Vertex array should have dimensions"*
            " (ni, nj, 3) where dimension 3 is x,y,z.")
        strs = zeros(size(vertices)[1]-1, size(vertices)[2]-1)
        return VortexLattice(vertices, strs)
    end

    function VortexLattice(geometry::Function;
        xs::Vector{Float64}=collect(-1:0.2:1), 
        ys::Vector{Float64}=collect(-1:0.2:1))
        @assert(hasmethod(geometry, (Float64, Float64)))
        
        d_fn = (x,y)->0.0
        return VortexLattice(geometry, d_fn;xs=xs, ys=ys)
    end

    function VortexLattice(geometry::Function, dipole_str::Function;
        xs::Vector{Float64}=collect(-1:0.2:1), 
        ys::Vector{Float64}=collect(-1:0.2:1))
        @assert(hasmethod(geometry, (Float64, Float64)))
        @assert(hasmethod(dipole_str, (Float64, Float64)))
        
        ni = length(xs)
        nj = length(ys)
        
        vl = new(zeros(ni, nj, 3), zeros(ni-1, nj-1))
        for i = 1 : ni
            for j  = 1 : nj
                vl.vertices[i, j, :] = geometry(xs[i], ys[j])
            end
        end
        for i = 1 : ni-1
            for j = 1 : nj-1
                x = (xs[i]+xs[i+1])/2
                y = (ys[j]+ys[j+1])/2
                vl.strengths[i, j] = dipole_str(x, y)
            end
        end
        return vl
    end
end


function check(a::VortexLattice; do_assert=false)
    str_finite = all(isfinite.(a.strengths))
    geo_finite = all(isfinite.(a.vertices))
    cor_size = true
    if prod(size(a.vertices)) > 0
        cor_size_i = (size(a.strengths)[1]+1) == size(a.vertices)[1]
        cor_size_j = (size(a.strengths)[2]+1) == size(a.vertices)[2]
        cor_size = cor_size_i && cor_size_j
    end
    if do_assert
        @assert(str_finite, 
            "VortexLattice constains non-finite ring strenghts.")
        @assert(geo_finite,
            "VortexLattice contains non-finite geometry coordinates")
        @assert(cor_size,
            "VortexLattice strength matrix and geometry matrix "*
            "are inconsistent sizes. Strength matrix is "*
            string(size(a.strengths))*" and geometry matrix is "*
            string(size(a.geometry))*".")
    end
    return str_finite && geo_finite && cor_size
end


#   Convert a VortexLattice to vectors representing individual filaments.
#   Outputs as Matrices: fstarts & fends, and Vector fstrengths
#   
#   Output ordering:
#   1st - filaments facing in the i-direction:
#       ->  Going by j index then i index
#   2nd - filaments facing in the j-direction:
#       ->  Going by j index then i index
function to_filaments(a::VortexLattice)
    nif = (size(a.vertices)[1]-1) * (size(a.vertices)[2]) # i-dir (streamwise)
    njf = (size(a.vertices)[1]) * (size(a.vertices)[2]-1) # j-dir (spanwise)
    nf = nif + njf
    fstarts = zeros(nf, 3)
    fends = zeros(nf, 3)
    fstrs = zeros(nf)
    acc = 1
    function rstr(i, j)
        if (i > 0) && (j > 0) && (i <= size(a.strengths)[1]) && (j <= size(a.strengths)[2])
            ret = a.strengths[i, j]
        else
            ret = 0
        end
        return ret
    end
    # (i --- i+1)
    for i = 1 : size(a.vertices)[1]-1
        for j = 1 : size(a.vertices)[2]
            fstarts[acc, :] = a.vertices[i, j, :]
            fends[acc, :] = a.vertices[i+1, j, :]
            fstrs[acc] = rstr(i, j) - rstr(i, j-1)
            acc += 1
        end
    end
    @assert(acc - 1 == nif)
    # (j --- j+1)
    for i = 1 : size(a.vertices)[1]
        for j = 1 : size(a.vertices)[2]-1
            fstarts[acc, :] = a.vertices[i, j, :]
            fends[acc, :] = a.vertices[i, j+1, :]
            fstrs[acc] = rstr(i-1, j) - rstr(i, j)
            acc += 1
        end
    end
    @assert(acc - 1 == nif + njf)
    return fstarts, fends, fstrs
end


function to_filaments(a::Vector{VortexLattice})
    fstarts = zeros(0, 3)
    fends = zeros(0, 3)
    fstrs = zeros(0)
    for i = 1 : length(a)
        tfs, tfe, tfstr = to_filaments(a[i])
        fstarts = vcat(fstarts, tfs)
        fends = vcat(fends, tfe)
        fstrs = vcat(fstrs, tfstr)
    end
    return fstarts, fends, fstrs
end


function to_vortex_particles(
    a::VortexLattice, 
    particle_separation::Float64)

    check(a; do_assert=true)

    fstarts, fends, fstrs = to_filaments(a)
    particle_locs, particle_strs = to_vortex_particles(
        fstarts, fends, fstrs,
        particle_separation)
    return particle_locs, particle_strs
end


function get_vertices(a::VortexLattice)
    ni = size(a.vertices)[1]
    nj = size(a.vertices)[2]
    nv = ni * nj
    verts = zeros(nv, 3)
    for i = 0 : nv-1
        verts[i+1, :] = a.vertices[Int64(i%ni+1), Int64(floor(i/ni)+1), :]
    end
    return verts
end


function set_vertices!(a::VortexLattice, vertices::Matrix{<:Real})
    @assert(size(vertices)[2] == 3)
    ani = size(a.vertices)[1]
    anj = size(a.vertices)[2]
    anv = ani * anj
    @assert(size(vertices)[1] == anv, "Trying to fit incorrect number of "*
        "vertices into array! Lattice has dimensions ("*string(ani)*
        ", "*string(anj)*") = length "*string(anv)*" but new verticies"*
        " matrix has "*string(size(vertices)[1])*" points.")
    for i = 0 : anv-1
        a.vertices[Int64(i%ani+1), Int64(floor(i/ani)+1), :] = vertices[i+1, :]
    end
    return
end


function get_strengths(a::VortexLattice)
    ni = size(a.strengths)[1]
    nj = size(a.strengths)[2]
    nv = ni * nj
    strengths = zeros(nv)
    for i = 1 : nv
        strengths[i] = a.strengths[i%ni+1, floor(i/ni)+1]
    end
    return strengths
end


function set_strengths!(a::VortexLattice, strengths::Vector{<:Real})
    ani = size(a.strength)[1]
    anj = size(a.strengths)[2]
    anv = ani * anj
    @assert(size(strengths)[1] == anv)
    for i = 1 : anv
        a.strengths[i%ani+1, floor(i/ani)+1] = strengths[i]
    end
    return
end


# The filament excluding the filaments that form the outer edge of the square.
function centre_filaments(
    a::VortexLattice
    ) :: Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}}

    nif = (size(a.vertices)[1]-1) * (size(a.vertices)[2]-2) # i-dir (streamwise)
    njf = (size(a.vertices)[1]-2) * (size(a.vertices)[2]-1) # j-dir (spanwise)
    nf = nif + njf
    fstarts = zeros(nf, 3)
    fends = zeros(nf, 3)
    fstrs = zeros(nf)
    acc = 1
    function rstr(i, j)
        if (i > 0) && (j > 0) && (i <= size(a.strengths)[1]) && (j <= size(a.strengths)[2])
            ret = a.strengths[i, j]
        else
            ret = 0
        end
        return ret
    end
    # (i --- i+1)
    for i = 1 : size(a.vertices)[1]-1
        for j = 2 : size(a.vertices)[2]-1
            fstarts[acc, :] = a.vertices[i, j, :]
            fends[acc, :] = a.vertices[i+1, j, :]
            fstrs[acc] = rstr(i, j) - rstr(i, j-1)
            acc += 1
        end
    end
    @assert(acc - 1 == nif)
    # (j --- j+1)
    for i = 2 : size(a.vertices)[1]-1
        for j = 1 : size(a.vertices)[2]-1
            fstarts[acc, :] = a.vertices[i, j, :]
            fends[acc, :] = a.vertices[i, j+1, :]
            fstrs[acc] = rstr(i-1, j) - rstr(i, j)
            acc += 1
        end
    end
    @assert(acc - 1 == nif + njf)
    return fstarts, fends, fstrs
end


function edge_filaments(a::VortexLattice;
    i_min::Bool=false, # The top row of fils.
    i_max::Bool=false, # The bottom row of fils.
    j_min::Bool=false, # The leftmost column of fils.
    j_max::Bool=false  # The rightmost column of fils.
    ) :: Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}}

    check(a; do_assert=true)

    ni, nj = size(a.strengths)
    nfils = 0
    nfils += i_min ? nj : 0
    nfils += i_max ? nj : 0
    nfils += j_min ? ni : 0
    nfils += j_max ? ni : 0
    acc = 1
    f_starts = zeros(nfils, 3)
    f_ends = zeros(nfils, 3)
    f_strs = zeros(nfils)
    if i_min
        f_starts[1:nj, :] = a.vertices[1, 1:nj, :]
        f_ends[1:nj, :] = a.vertices[1, 2:nj+1, :]
        f_strs[1:nj] = -a.strengths[1, 1:nj]
        acc += nj
    end
    if i_max
        rng = acc:acc+nj-1  
        f_starts[rng, :] = a.vertices[ni+1, 1:nj, :]
        f_ends[rng, :] = a.vertices[ni+1, 2:nj+1, :]
        f_strs[rng] = a.strengths[ni, 1:nj]
        acc += nj
    end
    if j_min
        rng = acc:acc+ni-1  
        f_starts[rng, :] = a.vertices[1:ni, 1, :]
        f_ends[rng, :] = a.vertices[2:ni+1, 1, :]
        f_strs[rng] = a.strengths[1:ni, 1]
        acc += ni
    end
    if j_max
        rng = acc:acc+ni-1  
        f_starts[rng, :] = a.vertices[1:ni, nj+1, :]
        f_ends[rng, :] = a.vertices[2:ni+1, nj+1, :]
        f_strs[rng] = -a.strengths[1:ni, nj]
        acc += ni   # I mean this is pointless, but meh.
    end

    return f_starts, f_ends, f_strs
end

#   Obtain the influence of rings on measurement points
#   Returns a matrix of nmes * nrings and a matrix mapping ring index to 
#   linear index.
function ring_influence_matrix(a::VortexLattice, 
    mespoints::Matrix{Float64}, mesnormals::Matrix{Float64}
    ) :: Tuple{Matrix{Float64}, Matrix{Int64}}

    @assert(size(mespoints)[2]==3, "Mesurement point matrix must be "*
        "number of points by 3.")
    @assert(size(mespoints)==size(mesnormals), "Measurement point "*
        "matrix and mesurement direction must be the same size.")

    # We're not using to_filaments here, to avoid linking the implementations
    # too closely.
    nif = (size(a.vertices)[1]-1) * (size(a.vertices)[2]) # i-dir (streamwise)
    njf = (size(a.vertices)[1]) * (size(a.vertices)[2]-1) # j-dir (spanwise)
    nf = nif + njf
    ni_s = size(a.strengths)[1]
    nj_s = size(a.strengths)[2]
    nrings = ni_s * nj_s
    ring_indices = reshape(collect(1:nrings), ni_s, nj_s)
    fil_mapping = zeros(nf, nrings)
    fstarts = zeros(nf, 3)
    fends = zeros(nf, 3)
    acc = 1
    # (i --- i+1)
    for i = 1 : size(a.strengths)[1]
        for j = 1 : size(a.strengths)[2]+1
            fstarts[acc, :] = a.vertices[i, j, :]
            fends[acc, :] = a.vertices[i+1, j, :]
            if j <= nj_s
                @assert(fil_mapping[acc, ring_indices[i,j]]==0)
                fil_mapping[acc, ring_indices[i,j]] = 1
            end
            if j > 1
                @assert(fil_mapping[acc,ring_indices[i,j-1]]==0)
                fil_mapping[acc,ring_indices[i,j-1]] = -1 
            end
            acc += 1
        end
    end
    @assert(acc - 1 == nif)
    # (j --- j+1)
    for i = 1 : size(a.strengths)[1]+1
        for j = 1 : size(a.strengths)[2]
            fstarts[acc, :] = a.vertices[i, j, :]
            fends[acc, :] = a.vertices[i, j+1, :]
            if i <= ni_s
                @assert(fil_mapping[acc,ring_indices[i,j]]==0)
                fil_mapping[acc,ring_indices[i,j]] = -1
            end
            if i > 1
                @assert(fil_mapping[acc,ring_indices[i-1,j]]==0)
                fil_mapping[acc,ring_indices[i-1,j]] = 1 
            end
            acc += 1
        end
    end
    @assert(acc - 1 == nif + njf)

    # Get the influence of individual filaments...
    vels = CVortex.filament_induced_velocity_influence_matrix(
        fstarts, fends, mespoints, mesnormals)
    vel_infs = vels * fil_mapping
    return vel_infs, ring_indices
end


# Make it easier to apply this function to 2d grid of 3d coords in
# the format used for the vortex lattice vertices. 
function ring_influence_matrix(a::VortexLattice, 
    mespoints::Array{Float64,3}, mesnormals::Array{Float64,3}
    ) :: Tuple{Matrix{Float64}, Matrix{Int64}, Matrix{Int64}}

    @assert(size(mespoints)[3]==3)
    @assert(size(mesnormals)==size(mespoints))
    # Reshape matrix so we only need one implementation of the dirty bits.
    ni, nj, ~ = size(mespoints)
    rspnts = reshape(mespoints, ni*nj, 3)
    rsnmls = reshape(mesnormals, ni*nj, 3)
    vel_inf, ring_indexing = ring_influence_matrix(a, rspnts, rsnmls)
    mes_pnt_indexing = reshape(collect(1:ni*nj), ni, nj)
    return vel_inf, ring_indexing, mes_pnt_indexing
end

#-- VortexLattice geometry ---------------------------------------------------

function extract_sublattice(a::VortexLattice, 
    i_start, i_end, j_start, j_end) :: VortexLattice

    check(a; do_assert=true)
    ni = extent_i(a)
    nj = extent_j(a)
    @assert(0 < i_start <= ni, "i_start is "*string(i_start)*" and "*
        "ni is "*string(ni)*".")
    @assert(0 < i_end <= ni, "i_end is "*string(i_end)*" and "*
        "ni is "*string(ni)*".")
    @assert(0 < j_start <= nj, "j_start is "*string(j_start)*" and "*
        "nj is "*string(nj)*".")
    @assert(0 < j_end <= nj, "j_end is "*string(j_end)*" and "*
        "nj is "*string(nj)*".")
    @assert(i_start <= i_end)
    @assert(j_start <= j_end)

    strs = a.strengths[i_start:i_end, j_start:j_end]
    vtxs = a.vertices[i_start:i_end+1, j_start:j_end+1, :]
    sublattice = VortexLattice(vtxs, strs)
    return sublattice
end


function ring_centres(a::VortexLattice) :: Array{Float64, 3}
    centres = (a.vertices[2:end,1:end-1,:] + a.vertices[1:end-1,2:end,:] + 
        a.vertices[2:end,2:end,:] + a.vertices[1:end-1,1:end-1,:]) ./ 4
    return centres
end


function ring_normals(a::VortexLattice) :: Array{Float64, 3}
    # Take the normals a the centres of linear quad elements.
    verts = a.vertices
    v00 = verts[1:end-1, 1:end-1,:]
    v01 = verts[1:end-1, 2:end,:]
    v10 = verts[2:end, 1:end-1,:]
    v11 = verts[2:end, 2:end,:]
    # Shape function derivatives gives direction in direction 0 and 1 
    # on element surface
    dXd0 = 0.25 .* (v00 + v01 - v10 - v11);
    dXd1 = 0.25 .* (-v00 + v01 - v10 + v11);
    # Now take cross product to obtain the normal.
    ret = zeros(size(v00))
    ret[:,:,1] = dXd0[:,:,2] .* dXd1[:,:,3] - dXd0[:,:,3] .* dXd1[:,:,2]
    ret[:,:,2] = dXd0[:,:,3] .* dXd1[:,:,1] - dXd0[:,:,1] .* dXd1[:,:,3]
    ret[:,:,3] = dXd0[:,:,1] .* dXd1[:,:,2] - dXd0[:,:,2] .* dXd1[:,:,1]
    # Now convert to unit vector.
    sx = size(ret)[1:2]
    for j = 1:sx[2]
        for i = 1:sx[1]
            ret[i,j,:] = ret[i,j,:] / sqrt(ret[i,j,1]^2+ret[i,j,2]^2+ret[i,j,3]^2)
        end
    end
    return ret
end

#-- TREATING VortexLattice like an array -------------------------------------

function extent_i(a::VortexLattice)
    return size(a.strengths)[1]
end

function extent_j(a::VortexLattice)
    return size(a.strengths)[2]
end


#   Add another row of rings to the vortex lattice.
#   Column is ended to the end. The index 'i' of this column is returned.
#   All new verticies and ring strengths are set to zero.
function add_row!(a::VortexLattice) :: Int64
    a.strengths = cat(a.strengths, zeros(size(a.strengths)[2]); dims=1)
    a.vertices = cat(a.vertices, zeros(1, size(a.vertices)[2], 3); dims=1)
    return size(a.strengths)[1]
end


#   Add another column of rings to the vortex lattice.
#   Column is ended to the end. The index 'j' of this column is returned.
#   All new verticies and ring strengths are set to zero.
function add_column!(a::VortexLattice) :: Int64
    a.strengths = cat(a.strengths, zeros(size(a.strengths)[1]); dims=2)
    a.vertices = cat(a.vertices, zeros(size(a.vertices)[1], 1, 3); dims=2)
    return size(a.strengths)[2]
end

#-- IO -----------------------------------------------------------------------

function to_vtk(a::VortexLattice,
    filename::String; translation::Vector{<:Real}=[0,0,0]) :: Nothing

    fstart, fend, fstr = to_filaments(a)
    to_vtk(VortexFilament, fstart, fend, fstr,
        filename; translation=translation)
    return
end

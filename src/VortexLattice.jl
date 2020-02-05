#
# VortexLattice.jl
#
# A square vortex lattice grid.
#
# Copyright HJA Bird 2019
#
#==============================================================================#

mutable struct VortexLattice
    vertices :: Array{Float64, 3}
    strengths :: Matrix{Float64}

    function VortexLattice(
        vertices::Array{Float64, 3}, strengths::Matrix{Float64})
        vl = new(vertices, strengths)
        @assert(check(vl))
        return vl
    end

    function VortexLattice(vertices::Array{Float64, 3})
        strs = zeros(size(vertices)[1]-1, size(vertices)[2]-1)
        return VortexLattice(vertices, strs)
    end

    function VortexLattice(geometry::Function, dipole_str::Function;
        xs::Vector{Float64}=[-1:0.2:1], ys=Vector{Float64}=[-1:0.2:1])
        @assert(hasmethod(geometry, (Float64, Float64)))
        @assert(hasmethod(dipole_str, (Float64, Float64)))
        
        ni = length(xs)
        nj = length(ys)
        
        vl = new(zeros(ni, nj, 3), zeros(ni-1, nj-1))
        for i = 1 : ni
            for j  = 1 : nj
                vl.vertices[i, j, :] = geometry(xs[i], ys[i])
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

function check(a::VortexLattice)
    str_finite = all(isfinite.(a.strengths))
    geo_finite = all(isfinite.(a.vertices))
    cor_size = true
    if prod(size(a.vertices)) > 0
        cor_size_i = (size(a.strengths)[1]+1) == size(a.vertices)[1]
        cor_size_j = (size(a.strengths)[2]+1) == size(a.vertices)[2]
        cor_size = cor_size_i && cor_size_j
    end
    return str_finite && geo_finite && cor_size
end

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

# The filament excluding the filaments that form the outer edge of the square.
function centre_filaments(a::VortexLattice)
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
    @assert(size(vertices)[1] == anv)
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

function extent_i(a::VortexLattice)
    return size(a.strengths)[1]
end

function extent_j(a::VortexLattice)
    return size(a.strengths)[2]
end

struct VortexFilament # We just want to be able to use the type
end

function to_vtk(::Type{VortexFilament},
    fil_starts::Matrix{<:Real}, fil_ends::Matrix{<:Real}, 
    fil_strs::Vector{<:Real}, filename::String;
    translation::Vector{<:Real}=[0,0,0])

    @assert(size(fil_starts)[2] == 3)
    @assert(size(fil_ends)[2] == 3)
    @assert(size(fil_strs)[1] == size(fil_starts)[1])
    @assert(size(fil_strs)[1] == size(fil_ends)[1])

    if prod(size(fil_strs))!= 0
        nfils = size(fil_starts)[1]
        points = vcat(fil_starts, fil_ends)
        points .+= translation'
        cells = Vector{WriteVTK.MeshCell}(undef, nfils)        
        celldata = fil_strs
        for i = 1 : nfils
            cells[i] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LINE, 
                [i, i+nfils])
        end
    else
        points = zeros(0, 3)
        cells = Vector{WriteVTK.MeshCell}(undef, 0)
        cell_str = zeros(0)
    end
    vtkfile = WriteVTK.vtk_grid(filename, points', cells)
    WriteVTK.vtk_cell_data(vtkfile, celldata, "Vorticity")
    WriteVTK.vtk_save(vtkfile)
end

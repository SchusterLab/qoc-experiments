"""
qocexperiments.jl - A file for common things shared in this repository.
"""

# paths / venv
WDIR = joinpath(@__DIR__, "../")
import Pkg
Pkg.activate(WDIR)

# imports
import Base
using HDF5
using LinearAlgebra
using Plots
using Printf
using RobotDynamics

# plotting configuration and constants
ENV["GKSwstype"] = "nul"
Plots.gr()
const DPI = 300
const MS_SMALL = 2
const MS_MED = 6
const ALPHA = 0.2

# types
@enum SaveType begin
    jl = 1
    samplejl = 2
    py = 3
end


@enum SolverType begin
    ilqr = 1
    alilqr = 2
    altro = 3
end


# methods
function generate_file_path(extension, file_name, path)
    # Ensure the path exists.
    mkpath(path)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for (_, _, files) in walkdir(path)
        for file_name_ in files
            if occursin("_$(file_name).$(extension)", file_name_)
                numeric_prefix = parse(Int, split(file_name_, "_")[1])
                max_numeric_prefix = max(numeric_prefix, max_numeric_prefix)
            end
        end
    end
    
    file_path = joinpath(path, "$(lpad(max_numeric_prefix + 1, 5, '0'))_$(file_name).$(extension)")
    return file_path
end


function latest_file_path(extension, file_name, path)
    max_numeric_prefix = -1
    for (_, _, files) in walkdir(path)
        for file_name_ in files
            if occursin("_$(file_name).$(extension)", file_name_)
                numeric_prefix = parse(Int, split(file_name_, "_")[1])
                max_numeric_prefix = max(numeric_prefix, max_numeric_prefix)
            end
        end
    end
    if max_numeric_prefix == -1
        file_path = nothing
    else
        file_path = joinpath(path, "$(lpad(max_numeric_prefix, 5, '0'))_$(file_name).$(extension)")
    end
    return file_path
end


"""
grab_controls - do some extraction of relevant controls
data for common h5 save formats
"""
function grab_controls(save_file_path)
    data = h5open(save_file_path, "r") do save_file
        save_type = SaveType(read(save_file, "save_type"))
        if save_type == jl
            cidx = read(save_file, "controls_idx")
            controls = read(save_file, "astates")[1:end - 1, cidx]
            evolution_time = read(save_file, "evolution_time")
            controls_dt_inv = haskey(save_file, "dt") ? read(save_file, "dt")^(-1) : DT_PREF_INV
        elseif save_type == samplejl
            controls = read(save_file, "controls_sample")[1:end - 1]
            evolution_time = read(save_file, "evolution_time_sample")
            controls_dt_inv = DT_PREF_INV
        elseif save_type == py
            controls = permutedims(read(save_file, "controls"), (2, 1))
            evolution_time = read(save_file, "evolution_time")
            controls_dt_inv = DT_PREF_INV
        end
        return (controls, controls_dt_inv, evolution_time)
    end

    return data
end


"""
read_save - Read all data from an h5 file into memory.
"""
function read_save(save_file_path)
    dict = h5open(save_file_path, "r") do save_file
        dict = Dict()
        for key in keys(save_file)
            dict[key] = read(save_file, key)
        end
        return dict
    end

    return dict
end


function plot_controls(save_file_paths, plot_file_path;
                       labels=nothing,
                       title="", colors=nothing, print_out=true,
                       legend=nothing, d2pi=false)
    fig = Plots.plot(dpi=DPI, title=title, legend=legend)
    for (i, save_file_path) in enumerate(save_file_paths)
        # Grab and prep data.
        (controls, controls_dt_inv, evolution_time) = grab_controls(save_file_path)
        (control_eval_count, control_count) = size(controls)
        control_eval_times = Array(0:control_eval_count - 1) / controls_dt_inv
        if d2pi
            controls = controls ./ (2 * pi)
        end
        
        # Plot.
        for j = 1:control_count
            label = isnothing(labels) ? nothing : labels[i][j]
            color = isnothing(colors) ? :auto : colors[i][j]
            Plots.plot!(control_eval_times, controls[:, j],
                        label=label, color=color)
        end
    end
    Plots.xlabel!("Time (ns)")
    Plots.ylabel!("Amplitude (GHz)")
    Plots.savefig(fig, plot_file_path)
    if print_out
        println("Plotted to $(plot_file_path)")
    end
    return
end


@inline show_nice(x) = show(IOContext(stdout), "text/plain", x)


@inline get_vec_iso(vec) = vcat(real(vec), imag(vec))


function get_vec_uniso(vec)
    n = size(vec)[1]
    nby2 = Integer(n/2)
    i1 = 1:nby2
    i2 = (nby2 + 1):n
    return vec[i1] + 1im * vec[i2]
end


function get_mat_iso(mat)
    mat_r = real(mat)
    mat_i = imag(mat)
    return vcat(hcat(mat_r, -mat_i),
                hcat(mat_i,  mat_r))
end


function get_mat_uniso(mat)
    n = size(mat, 1)
    nby2 = Integer(n/2)
    i1 = 1:nby2
    i2 = (nby2 + 1):n
    return mat[i1, i1] + mat[i2, i1]
end


@inline get_vec_viso(mat) = reshape(mat, length(mat))


@inline get_vec_unviso(vec) = reshape(vec, (Int(sqrt(length(vec))), Int(sqrt(length(vec)))))

# Adapted from implementation in StaticArrays
# https://github.com/JuliaArrays/StaticArrays.jl/blob/master/src/expm.jl
# Adapted from implementation in Base; algorithm from
# Higham, "Functions of Matrices: Theory and Computation", SIAM, 2008
function Base.exp(_A::AbstractMatrix{T}) where T
    S = typeof((zero(T)*zero(T) + zero(T)*zero(T))/one(T))
    A = S.(_A)
    # omitted: matrix balancing, i.e., LAPACK.gebal!
    nA = maximum(sum(abs.(A); dims=1))    # marginally more performant than norm(A, 1)
    ## For sufficiently small nA, use lower order Padé-Approximations
    if (nA <= 2.1)
        A2 = A*A
        if nA > 0.95
            U = @evalpoly(A2, S(8821612800)*I, S(302702400)*I, S(2162160)*I, S(3960)*I, S(1)*I)
            U = A*U
            V = @evalpoly(A2, S(17643225600)*I, S(2075673600)*I, S(30270240)*I, S(110880)*I, S(90)*I)
        elseif nA > 0.25
            U = @evalpoly(A2, S(8648640)*I, S(277200)*I, S(1512)*I, S(1)*I)
            U = A*U
            V = @evalpoly(A2, S(17297280)*I, S(1995840)*I, S(25200)*I, S(56)*I)
        elseif nA > 0.015
            U = @evalpoly(A2, S(15120)*I, S(420)*I, S(1)*I)
            U = A*U
            V = @evalpoly(A2, S(30240)*I, S(3360)*I, S(30)*I)
        else
            U = @evalpoly(A2, S(60)*I, S(1)*I)
            U = A*U
            V = @evalpoly(A2, S(120)*I, S(12)*I)
        end
        expA = (V - U) \ (V + U)
    else
        s  = log2(nA/5.4)               # power of 2 later reversed by squaring
        if s > 0
            si = ceil(Int,s)
            A = A / S(2^si)
        end

        A2 = A*A
        A4 = A2*A2
        A6 = A2*A4

        U = A6*(S(1)*A6 + S(16380)*A4 + S(40840800)*A2) +
            (S(33522128640)*A6 + S(10559470521600)*A4 + S(1187353796428800)*A2) +
            S(32382376266240000)*I
        U = A*U
        V = A6*(S(182)*A6 + S(960960)*A4 + S(1323241920)*A2) +
            (S(670442572800)*A6 + S(129060195264000)*A4 + S(7771770303897600)*A2) +
            S(64764752532480000)*I
        expA = (V - U) \ (V + U)

        if s > 0            # squaring to reverse dividing by power of 2
            for t=1:si
                expA = expA*expA
            end
        end
    end

    expA
end

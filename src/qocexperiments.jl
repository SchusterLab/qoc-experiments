"""
qocexperiments.jl - A file for common things shared in this repository.
"""

using HDF5
using Pkg
using Plots
using Printf
using RobotDynamics

# set repository path and activate environment
WDIR = get(ENV, "QOC_EXPERIMENTS_PATH", "../")
Pkg.activate(WDIR)

# plotting configuration and constants
ENV["GKSwstype"] = "nul"
Plots.gr()
DPI = 300
MS_SMALL = 2
ALPHA = 0.2

# types
@enum SolverType begin
    ilqr = 1
    alilqr = 2
    altro = 3
end


@enum IntegratorType begin
    rk2 = 1
    rk3 = 2
    rk4 = 3
end


const IT_RDI = Dict(
    rk2 => RobotDynamics.RK2,
    rk3 => RobotDynamics.RK3,
    rk4 => RobotDynamics.RK4,
)


"""
common file formats
"""
@enum SaveType begin
    jl = 1
    samplejl = 2
    py = 3
end


# methods
"""
generate_file_path - generate a save file path like
    "save_path/XXXXX_save_file_name.extension" where X
    is an integer.
"""
function generate_file_path(extension, save_file_name, save_path)
    # Ensure the path exists.
    mkpath(save_path)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for (_, _, files) in walkdir(save_path)
        for file_name in files
            if occursin("_$(save_file_name).$(extension)", file_name)
                max_numeric_prefix = max(parse(Int, split(file_name, "_")[1]))
            end
        end
    end

    save_file_name = "_$(save_file_name).$(extension)"
    save_file_name = @sprintf("%05d%s", max_numeric_prefix + 1, save_file_name)

    return joinpath(save_path, save_file_name)
end


"""
read_save - Read all data from an h5 file into memory.
"""
function read_save(save_file_path)
    dict = h5open(save_file_path, "r") do save_file
        dict = Dict()
        for key in names(save_file)
            dict[key] = read(save_file, key)
        end
        return dict
    end

    return dict
end


"""
grab_controls - do some extraction of relevant controls
data for common h5 save formats
"""
function grab_controls(save_file_path; save_type=jl)
    data = h5open(save_file_path, "r") do save_file
        if save_type == jl
            cidx = read(save_file, "controls_idx")
            controls = read(save_file, "astates")[:, cidx]
            evolution_time = read(save_file, "evolution_time")
        elseif save_type == samplejl
            controls = read(save_file, "controls_sample")
            evolution_time = read(save_file, "evolution_time_sample")
        elseif save_type == py
            controls = permutedims(read(save_file, "controls"), (2, 1))
            evolution_time = read(save_file, "evolution_time")
        end
        return (controls, evolution_time)
    end

    return data
end


function plot_controls(save_file_paths, plot_file_path;
                       save_types=[jl,], labels=nothing,
                       title="", colors=nothing, print_out=true,
                       legend=nothing)
    fig = Plots.plot(dpi=DPI, title=title, legend=legend)
    for (i, save_file_path) in enumerate(save_file_paths)
        # Grab and prep data.
        (controls, evolution_time) = grab_controls(save_file_path; save_type=save_types[i])
        controls = controls ./ (2 * pi)
        (control_eval_count, control_count) = size(controls)
        control_eval_times = Array(1:1:control_eval_count) * DT_PREF
        
        # Plot.
        for j = 1:control_count
            if labels == nothing
                label = nothing
            else
                label = labels[i][j]
            end
            if colors == nothing
                color = nothing
            else
                color = colors[i][j]
            end
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


"""
pretty-print a tensor
"""
show_nice(x) = show(IOContext(stdout), "text/plain", x)


function get_vec_iso(vec)
    return [real(vec);
            imag(vec)]
end


function get_mat_iso(mat)
    len = size(mat)[1]
    mat_r = real(mat)
    mat_i = imag(mat)
    return [mat_r -mat_i;
            mat_i  mat_r]
end


function fidelity_vec_iso(v1_, v2_)
    n = size(v1_)[1]
    nby2 = Integer(n/2)
    i1 = 1:nby2
    i2 = (nby2 + 1):n
    v1 = v1_[i1] + 1im * v1_[i2]
    v2 = v2_[i1] + 1im * v2_[i2]
    return abs(v1'v2)^2
end


"""
See e.q. 9.71 in [0]

[0] Nielsen, M. A., & Chuang, I. (2002).
    Quantum computation and quantum information.
"""
function fidelity_mat_iso(m1_, m2_)
    n = size(m1_)[1]
    nby2 = Integer(n/2)
    i1 = 1: nby2
    i2 = (nby2 + 1):n
    m1 = m1_[i1, i1] + 1im * m1_[i2, i1]
    m2 = m2_[i1, i1] + 1im * m2_[i2, i1]
    sqrt_m1 = sqrt(Hermitian(m1))
    sqrt_m2 = sqrt(Hermitian(m2))
    return tr(sqrt_m1 * sqrt_m2)^2
end

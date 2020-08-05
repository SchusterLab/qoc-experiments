"""
common.jl - A common file for the vslq-spinspin directory.
"""

using Dates
using Dierckx
using DifferentialEquations
using HDF5
using LinearAlgebra
using Plots
using Printf
using Random
using StaticArrays

### COMMON ###

# simulation constants
DT_PREF = 1e-3
DT_PREF_INV = 1e3

# plotting configuration and constants
ENV["GKSwstype"] = "nul"
Plots.gr()
DPI = 300
MS_SMALL = 2
ALPHA = 0.2

# other constants
DEQJL_MAXITERS = 1e10
DEQJL_ADAPTIVE = false

@enum DynamicsType begin
    schroed = 1
    lindbladnodis = 2
    lindbladdis = 3
end


@enum GateType begin
    zpiby2 = 1
    ypiby2 = 2
    xpiby2 = 3
end

@enum SaveType begin
    jl = 1
    samplejl = 2
    py = 3
end


# methods
"""
generate_save_file_path - generate a save file path like
    "save_path/XXXXX_save_file_name.extension" where X
    is an integer.
"""
function generate_save_file_path(extension, save_file_name, save_path)
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
sample_controls - Sample controls and d2controls_dt2
on the preferred time axis using a spline.
"""
function sample_controls(save_file_path; dt=DT_PREF, dt_inv=DT_PREF_INV,
                         plot=false, plot_file_path=nothing)
    # Grab data to sample from.
    save = read_save(save_file_path)
    controls = save["astates"][1:end - 1, (save["controls_idx"])]
    d2controls_dt2 = save["acontrols"][1:end, save["d2controls_dt2_idx"]]
    (control_knot_count, control_count) = size(controls)
    dts = save["acontrols"][1:end, save["dt_idx"]]
    time_axis = [0; cumsum(dts, dims=1)[1:end-1]]

    # Construct time axis to sample over.
    final_time_sample = sum(dts)
    knot_count_sample = Int(floor(final_time_sample * dt_inv))
    # The last control should be DT_PREF before final_time_sample.
    time_axis_sample = Array(0:1:knot_count_sample - 1) * dt

    # Sample time_axis_sample via spline.
    controls_sample = zeros(knot_count_sample, control_count)
    d2controls_dt2_sample = zeros(knot_count_sample, control_count)
    for i = 1:control_count
        controls_spline = Spline1D(time_axis, controls[:, i])
        controls_sample[:, i] = map(controls_spline, time_axis_sample)
        d2controls_dt2_spline = Spline1D(time_axis, d2controls_dt2[:, i])
        d2controls_dt2_sample[:, i] = map(d2controls_dt2_spline, time_axis_sample)
    end

    # Plot.
    if plot
        fig = Plots.plot(dpi=DPI)
        Plots.scatter!(time_axis, controls[:, 1], label="controls data", markersize=MS_SMALL, alpha=ALPHA)
        Plots.scatter!(time_axis_sample, controls_sample[:, 1], label="controls fit",
                       markersize=MS_SMALL, alpha=ALPHA)
        Plots.scatter!(time_axis, d2controls_dt2[:, 1], label="d2_controls_dt2 data")
        Plots.scatter!(time_axis_sample, d2controls_dt2_sample[:, 1], label="d2_controls_dt2 fit")
        Plots.xlabel!("Time (ns)")
        Plots.ylabel!("Amplitude (GHz)")
        Plots.savefig(fig, plot_file_path)
        println("Plotted to $(plot_file_path)")
    end
    return (controls_sample, d2controls_dt2_sample, final_time_sample)
end


function dynamics_schroed_deqjl(state, (controls, control_knot_count, dt_inv), t)
    knot_point = (Int(floor(t * dt_inv)) % control_knot_count) + 1
    neg_i_h = (
        NEGI_H0_ISO
        + controls[knot_point][1] * NEGI_H1_ISO
    )
    return (
        neg_i_h * state
    )
end


function dynamics_lindblad_deqjl(density, (controls, control_knot_count, dt_inv), t)
    knot_point = (Int(floor(t * dt_inv)) % control_knot_count) + 1
    neg_i_h = (
        NEGI_H0_ISO
        + controls[knot_point][1] * NEGI_H1_ISO
    )
    return (
        neg_i_h * density - density * neg_i_h
    )
end


function dynamics_lindblad_nodis_deqjl(density, (controls, control_knot_count, dt_inv), t)
    knot_point = (Int(floor(t * dt_inv)) % control_knot_count) + 1
    neg_i_h = (
        NEGI_H0_ISO
        + controls[knot_point][1] * NEGI_H1_ISO
    )
    return (
        neg_i_h * density - density * neg_i_h
    )
end


fidelity_mat(m1, m2) = abs(tr(m1' * m2)) / abs(tr(m2' * m2))


function gen_rand_state_iso(;seed=0)
    Random.seed!(seed)
    state = rand(HILBERT_SIZE_NOISO) + 1im * rand(HILBERT_SIZE_NOISO)
    state_normalized = state ./ sqrt(abs(state'state))
    return SVector{HILBERT_SIZE_ISO}(
        [real(state_normalized); imag(state_normalized)]
    )
end


function gen_rand_density_iso(;seed=0)
    Random.seed!(seed)
    state = rand(HILBERT_SIZE_NOISO) + 1im * rand(HILBERT_SIZE_NOISO)        
    density = (state * state') / abs(state' * state)
    density_r = real(density)
    density_i = imag(density)
    density_iso = SMatrix{HILBERT_SIZE_ISO, HILBERT_SIZE_ISO}([
        density_r -density_i;
        density_i density_r;
    ])
    return density_iso
end


"""
run_sim_deqjl - Apply a gate multiple times and measure the fidelity
after each application. Save the output.

Arguments:
save_file_path :: String - The file path to grab the controls from
"""
function run_sim_deqjl(
    gate_count,
    save_file_path;
    controls_dt_inv=DT_PREF_INV,
    deqjl_adaptive=false, dynamics_type=schroed,
    dt=DT_PREF, print_final=false, save=true, save_type=jl, seed=0,
    solver=DifferentialEquations.Vern9)
    start_time = Dates.now()
    # grab
    (controls, gate_time) = grab_controls(save_file_path; save_type=save_type)
    gate_knot_count = Int(floor(gate_time * controls_dt_inv))
    gate_times = Array(0:1:gate_count) * gate_time
    save_times = Array(0:1:gate_knot_count) * dt
    
    # integrate
    if dynamics_type == schroed
        f = dynamics_schroed_deqjl
    elseif dynamics_type == lindbladnodis
        f = dynamics_lindblad_nodis_deqjl
    else
        f = dynamics_lindblad_deqjl
    end
    if dynamics_type == schroed
        x0 = gen_rand_state_iso(;seed=seed)
    else
        x0 = gen_rand_density_iso(;seed=seed)
    end
    tspan = (0., gate_time * gate_count)
    p = (controls, gate_knot_count, controls_dt_inv)
    prob = ODEProblem(f, x0, tspan, p)
    result = solve(prob, solver(), dt=dt, saveat=save_times,
                   maxiters=DEQJL_MAXITERS, adaptive=deqjl_adaptive)
    
    states_ = result.u
    return states_

    # # Compute the fidelities.
    # # All of the gates we consider are 4-cyclic.
    # densities = zeros(gate_count + 1, STATE_SIZE_ISO, STATE_SIZE_ISO)
    # fidelities = zeros(gate_count + 1)
    # g1 = GT_GATE[gate_type]
    # g2 = g1^2
    # g3 = g1^3
    # id0 = initial_density
    # id1 = g1 * id0 * g1'
    # id2 = g2 * id0 * g2'
    # id3 = g3 * id0 * g3'
    # target_dag = id0_dag = id0'
    # id1_dag = id1'
    # id2_dag = id2'
    # id3_dag = id3'
    # target_fnorm = id0_fnorm = abs(tr(id0_dag * id0))
    # id1_fnorm = abs(tr(id1_dag * id1))
    # id2_fnorm = abs(tr(id2_dag * id2))
    # id3_fnorm = abs(tr(id3_dag * id3))
    # # Compute the fidelity after each gate.
    # for i = 1:gate_count + 1
    #     densities[i, :, :] = density = result.u[i]
    #     # 1-indexing means we are 1 ahead for modulo arithmetic.
    #     i_eff = i - 1
    #     if i_eff % 4 == 0
    #         target_dag = id0_dag
    #         target_fnorm = id0_fnorm
    #     elseif i_eff % 4 == 1
    #         target_dag = id1_dag
    #         target_fnorm = id1_fnorm
    #     elseif i_eff % 4 == 2
    #         target_dag = id2_dag
    #         target_fnorm = id2_fnorm
    #     elseif i_eff % 4 == 3
    #         target_dag = id3_dag
    #         target_fnorm = id3_fnorm
    #     end
    #     fidelities[i] = abs(tr(target_dag * density)) / target_fnorm
    #     # println("fidelity\n$(fidelities[i])")
    #     # println("density")
    #     # show_nice(density)
    #     # println("")
    #     # println("target")
    #     # show_nice(target_dag')
    #     # println("")
    # end
    # end_time = Dates.now()
    # run_time = end_time - start_time
    # if print_final
    #     println("fidelities[$(gate_count)]: $(fidelities[end])")
    # end

    # # Save the data.
    # experiment_name = split(save_file_path, "/")[end - 1]
    # save_path = dirname(save_file_path)
    # data_file_path = nothing
    # if save
    #     data_file_path = generate_save_file_path("h5", experiment_name, save_path)
    #     h5open(data_file_path, "w") do data_file
    #         write(data_file, "dynamics_type", Integer(dynamics_type))
    #         write(data_file, "gate_count", gate_count)
    #         write(data_file, "gate_time", gate_time)
    #         write(data_file, "gate_type", Integer(gate_type))
    #         write(data_file, "save_file_path", save_file_path)
    #         write(data_file, "seed", seed)
    #         write(data_file, "densities", densities)
    #         write(data_file, "fidelities", fidelities)
    #         write(data_file, "run_time", string(run_time))
    #     end
    #     println("Saved simulation to $(data_file_path)")
    # end
    # return data_file_path
end


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


### VSLQ SPIN SPIN ###

# Define experimental constants.
OMEGA_1 = 3.5 #GHz
OMEGA_2 = 4.2 #GHz
G = -2.1e-3 #GHz
MAX_AMP_1 = 3e-2 #GHz
MAX_AMP_2 = 3e-2 #GHz

# Define the system.
# ISO means the object is defined in the complex to real isomorphism.
# NEG_I is the negative complex unit
HILBERT_SIZE_NOISO = 4
HILBERT_SIZE_ISO = 8
_II_ISO = get_mat_iso(kron(I(2), I(2)))
II_ISO_1 = SizedVector{HILBERT_SIZE_ISO}(_II_ISO[:, 1])
II_ISO_2 = SizedVector{HILBERT_SIZE_ISO}(_II_ISO[:, 2])
II_ISO_3 = SizedVector{HILBERT_SIZE_ISO}(_II_ISO[:, 3])
II_ISO_4 = SizedVector{HILBERT_SIZE_ISO}(_II_ISO[:, 4])
XPIBY2 = [1 -1im;
          -1im 1] ./ sqrt(2)
_XPIBY2I_ISO = get_mat_iso(kron(XPIBY2, I(2)))
XPIBY2I_ISO_1 = SizedVector{HILBERT_SIZE_ISO}(_XPIBY2I_ISO[:, 1])
XPIBY2I_ISO_2 = SizedVector{HILBERT_SIZE_ISO}(_XPIBY2I_ISO[:, 2])
XPIBY2I_ISO_3 = SizedVector{HILBERT_SIZE_ISO}(_XPIBY2I_ISO[:, 3])
XPIBY2I_ISO_4 = SizedVector{HILBERT_SIZE_ISO}(_XPIBY2I_ISO[:, 4])
_IXPIBY2_ISO = get_mat_iso(kron(I(2), XPIBY2))
IXPIBY2_ISO_1 = SizedVector{HILBERT_SIZE_ISO}(_IXPIBY2_ISO[:, 1])
IXPIBY2_ISO_2 = SizedVector{HILBERT_SIZE_ISO}(_IXPIBY2_ISO[:, 2])
IXPIBY2_ISO_3 = SizedVector{HILBERT_SIZE_ISO}(_IXPIBY2_ISO[:, 3])
IXPIBY2_ISO_4 = SizedVector{HILBERT_SIZE_ISO}(_IXPIBY2_ISO[:, 4])

# SIGMA_X, SIGMA_Z are the X and Z Pauli matrices.
SIGMA_X = [0 1;
           1 0]
SIGMA_Z = [1 0;
           0 -1]
NEGI_ZI_ISO = SizedMatrix{HILBERT_SIZE_ISO, HILBERT_SIZE_ISO}(
    get_mat_iso(kron(-1im * SIGMA_Z, I(2)))
)
NEGI_IZ_ISO = SizedMatrix{HILBERT_SIZE_ISO, HILBERT_SIZE_ISO}(
    get_mat_iso(kron(-1im * I(2), SIGMA_Z))
)
NEGI_ZZ_ISO = SizedMatrix{HILBERT_SIZE_ISO, HILBERT_SIZE_ISO}(
    get_mat_iso(kron(-1im * SIGMA_Z, SIGMA_Z))
)
NEGI_XI_ISO = SizedMatrix{HILBERT_SIZE_ISO, HILBERT_SIZE_ISO}(
    get_mat_iso(kron(-1im * SIGMA_X, I(2)))
)
NEGI_IX_ISO = SizedMatrix{HILBERT_SIZE_ISO, HILBERT_SIZE_ISO}(
    get_mat_iso(kron(-1im * I(2), SIGMA_X))
)
NEGI_H0_ISO = 2 * pi * (
    -(OMEGA_1 / 2 + G) * NEGI_ZI_ISO
    -(OMEGA_2 / 2 + G) * NEGI_IZ_ISO
    + G * NEGI_ZZ_ISO
)
NEGI_H1_ISO = 2 * pi * NEGI_XI_ISO
NEGI_H2_ISO = 2 * pi * NEGI_IX_ISO

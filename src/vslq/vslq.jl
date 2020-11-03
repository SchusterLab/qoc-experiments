"""
vslq.jl - A common file for the vslq directory.
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
using Statistics

WDIR = get(ENV, "QOC_EXPERIMENTS_PATH", "../../")
include(joinpath(WDIR, "src", "qocexperiments.jl"))

### COMMON ###

## CONSTANTS ##

# paths
const VSLQ_OUT_PATH = joinpath(WDIR, "out", "vslq")

# simulation constants
const DT_PREF = 1e-2
const DT_PREF_INV = 1e2

# other constants
const DEQJL_MAXITERS = 1e10
const DEQJL_ADAPTIVE = false


## TYPES ##
@enum DynamicsType begin
    schroed = 1
    lindbladnodis = 2
    lindbladdis = 3
    empty = 4
end

@enum StateType begin
    st_state = 1
    st_density = 2
end

@enum GateType begin
    zpiby2 = 1
    ypiby2 = 2
    xpiby2 = 3
end

struct SimParams
    controls :: Array{Float64, 2}
    control_knot_count :: Int64
    controls_dt_inv :: Int64
    negi_h0 :: StaticMatrix
    sim_dt_inv :: Int64
end


## SYSTEM ##
# Define experimental constants.
const OMEGA_1 = 3.5 #GHz
const OMEGA_2 = 4.2 #GHz
const GCOUPLE = -2.1e-3 #GHz
const MAX_AMP = 3e-2 #GHz

# Define the system.
# ISO means the object is defined in the complex to real isomorphism.
# NEGI is the negative complex unit.
const HDIM = 4
const HDIM_ISO = 8
const _II_ISO = get_mat_iso(kron(I(2), I(2)))
const II_ISO_1 = SizedVector{HDIM_ISO}(_II_ISO[:, 1])
const II_ISO_2 = SizedVector{HDIM_ISO}(_II_ISO[:, 2])
const II_ISO_3 = SizedVector{HDIM_ISO}(_II_ISO[:, 3])
const II_ISO_4 = SizedVector{HDIM_ISO}(_II_ISO[:, 4])
const XPIBY2 = [1 -1im;
                -1im 1] ./ sqrt(2)
const _XPIBY2I_ISO = get_mat_iso(kron(XPIBY2, I(2)))
const XPIBY2I_ISO = SizedMatrix{HDIM_ISO, HDIM_ISO}(_XPIBY2I_ISO)
const XPIBY2I_ISO_1 = SizedVector{HDIM_ISO}(_XPIBY2I_ISO[:, 1])
const XPIBY2I_ISO_2 = SizedVector{HDIM_ISO}(_XPIBY2I_ISO[:, 2])
const XPIBY2I_ISO_3 = SizedVector{HDIM_ISO}(_XPIBY2I_ISO[:, 3])
const XPIBY2I_ISO_4 = SizedVector{HDIM_ISO}(_XPIBY2I_ISO[:, 4])
const _IXPIBY2_ISO = get_mat_iso(kron(I(2), XPIBY2))
const IXPIBY2_ISO_1 = SizedVector{HDIM_ISO}(_IXPIBY2_ISO[:, 1])
const IXPIBY2_ISO_2 = SizedVector{HDIM_ISO}(_IXPIBY2_ISO[:, 2])
const IXPIBY2_ISO_3 = SizedVector{HDIM_ISO}(_IXPIBY2_ISO[:, 3])
const IXPIBY2_ISO_4 = SizedVector{HDIM_ISO}(_IXPIBY2_ISO[:, 4])

# SIGMA_X, SIGMA_Z are the X and Z Pauli matrices.
const SIGMA_X = [0 1;
                 1 0]
const SIGMA_Z = [1 0;
                 0 -1]
const NEGI_ZI_ISO = SizedMatrix{HDIM_ISO, HDIM_ISO}(
    get_mat_iso(kron(-1im * SIGMA_Z, I(2)))
)
const NEGI_IZ_ISO = SizedMatrix{HDIM_ISO, HDIM_ISO}(
    get_mat_iso(kron(-1im * I(2), SIGMA_Z))
)
const NEGI_ZZ_ISO = SizedMatrix{HDIM_ISO, HDIM_ISO}(
    get_mat_iso(kron(-1im * SIGMA_Z, SIGMA_Z))
)
const NEGI_XI_ISO = SizedMatrix{HDIM_ISO, HDIM_ISO}(
    get_mat_iso(kron(-1im * SIGMA_X, I(2)))
)
const NEGI_IX_ISO = SizedMatrix{HDIM_ISO, HDIM_ISO}(
    get_mat_iso(kron(-1im * I(2), SIGMA_X))
)
# H0 = -(ω1 / 2 + g) ZI -(ω2 / 2 + g) IZ + g ZZ
const NEGI_H0_ISO = 2 * pi * (
    -(OMEGA_1 / 2 + GCOUPLE) * NEGI_ZI_ISO
    -(OMEGA_2 / 2 + GCOUPLE) * NEGI_IZ_ISO
    + GCOUPLE * NEGI_ZZ_ISO
)
# H1 = a1(t) XI
const NEGI_H1_ISO = 2 * pi * NEGI_XI_ISO
# H2 = a2(t) IX
const NEGI_H2_ISO = 2 * pi * NEGI_IX_ISO


## METHODS ##

function dynamics_schroed_deqjl(state::StaticVector, params::SimParams, t::Float64)
    knot_point = (Int(floor(t * params.control_dt_inv)) % params.control_knot_count) + 1
    negi_h = (
        NEGI_H0_ISO
        + params.controls[knot_point][1] * NEGI_H1_ISO
    )
    return (
        negi_h * state
    )
end


const H_EMPTY = NEGI_H0_ISO + MAX_AMP * (NEGI_H1_ISO + NEGI_H2_ISO)
@inline dynamics_empty_deqjl(state::StaticVector, params::SimParams, t::Float64) = (
    H_EMPTY * state 
)


function dynamics_lindbladnodis_deqjl(density::StaticMatrix, params::SimParams, t::Float64)
    knot_point = (Int(floor(t * params.control_dt_inv)) % params.control_knot_count) + 1
    neg_i_h = (
        NEGI_H0_ISO
        + params.controls[knot_point][1] * NEGI_H1_ISO
    )
    return (
        neg_i_h * density - density * neg_i_h
    )
end


const DT_ST = Dict(
    schroed => st_state,
    lindbladnodis => st_density,
    empty => st_state,
)

const DT_GTM = Dict(
    empty => 30.
)

const DT_EN = Dict(
    empty => "spinspin1"
)

const GT_GATE = Dict(
    xpiby2 => XPIBY2I_ISO
)

const DT_DYN = Dict(
    schroed => dynamics_schroed_deqjl,
    lindbladnodis => dynamics_lindbladnodis_deqjl,
    empty => dynamics_empty_deqjl,
)


function gen_rand_state_iso(;seed=0)
    if seed == 0
        state = [1,
                 0,
                 1,
                 0] / sqrt(2)
    else
        Random.seed!(seed)
        state = rand(HDIM) + 1im * rand(HDIM)
    end
    return SizedVector{HDIM_ISO}(
        [real(state); imag(state)]
    )
end


function gen_rand_density_iso(;seed=0)
    if seed == 0
        state = [1, 1] / sqrt(2)
    else
        Random.seed!(seed)
        state = rand(HDIM) + 1im * rand(HDIM)        
    end
    density = (state * state') / abs(state' * state)
    density_r = real(density)
    density_i = imag(density)
    density_iso = SizedMatrix{HDIM_ISO, HDIM_ISO}([
        density_r -density_i;
        density_i density_r;
    ])
    return density_iso
end


"""
run_sim_deqjl - Measure the fidelity of a gate.

Arguments:
save_file_path :: String - The file path to grab the controls from
"""
function run_sim_deqjl(
    gate_count, gate_type;
    save_file_path=nothing,
    controls_dt_inv=DT_PREF_INV,
    adaptive=DEQJL_ADAPTIVE, dynamics_type=lindbladnodis,
    dt=DT_PREF, save=true, save_type=jl, seed=0,
    solver=DifferentialEquations.Vern9, print_final=false,
    negi_h0=NEGI_H0_ISO)
    start_time = Dates.now()
    # grab
    if isnothing(save_file_path)
        controls = Array{Float64, 2}([0 0])
        control_knot_count = 0
        gate_time = DT_GTM[dynamics_type]
    else
        (controls, gate_time) = grab_controls(save_file_path; save_type=save_type)
        control_knot_count = Int(floor(gate_time * controls_dt_inv))
    end
    dt_inv = 1 / dt
    evolution_time = gate_time * gate_count
    knot_count = Int(ceil(evolution_time * dt_inv))
    # save_times = Array(0:1:gate_count) * gate_time
    save_times = Array(1:knot_count) * dt
    
    # integrate
    dynamics = DT_DYN[dynamics_type]
    state_type = DT_ST[dynamics_type]
    if state_type == st_state
        initial_state = gen_rand_state_iso(;seed=seed)
    elseif state_type == st_density
        initial_state =  gen_rand_density_iso(;seed=seed)
    end
    tspan = (0., evolution_time)
    params = SimParams(controls, control_knot_count, controls_dt_inv, negi_h0,
                       dt_inv)
    prob = ODEProblem(dynamics, initial_state, tspan, params)
    result = solve(prob, solver(), dt=dt, saveat=save_times,
                   maxiters=DEQJL_MAXITERS, adaptive=adaptive)
    states = result.u
    final_state = result.u[end]
    
    time = 0.
    state = initial_state
    if state_type == st_state
        states = zeros(knot_count, HDIM_ISO)
    elseif state_type == st_density
        states = zeros(knot_count, HDIM_ISO, HDIM_ISO)
    end
    for i = 1:knot_count
        time = time + dt
        state = rk6_step(dynamics, state, params, time, dt)
        if state_type == st_state
            states[i, :] = state
        else
            states[i, :, :] = state
        end
    end
    final_state = state

    # obtain fidelity
    gate = GT_GATE[gate_type]

    if state_type == st_state
        target_state = gate * initial_state
        fidelity = abs(fidelity_vec_iso(final_state, target_state))
    elseif state_type == st_density
        target_state = gate * initial_state * gate'
        fidelity = abs(fidelity_mat_iso(final_state, target_state))
    end
    fidelities = [1., fidelity]
    end_time = Dates.now()
    run_time = end_time - start_time


    norms = [states[i]'states[i] for i = 1:size(states)[1]]
    # norms = [states[i, :]'states[i, :] for i = 1:size(states)[1]]
    min_norm = minimum(norms)
    max_norm = maximum(norms)
    avg_norm = mean(norms)
    println("min_n: $(min_norm), max_norm: $(max_norm), avg_norm: $(avg_norm)")
    return

    if print_final
        println("fidelity: $(fidelity)")
        println("final_state")
        show_nice(final_state)
        println("\ntarget_state")
        show_nice(target_state)
        println("")
    end

    # Save the data.
    experiment_name = save_path = nothing
    if isnothing(save_file_path)
        experiment_name = DT_EN[dynamics_type]
        save_path = joinpath(VSLQ_OUT_PATH, experiment_name)
    else
        experiment_name = split(save_file_path, "/")[end - 1]
        save_path = dirname(save_file_path)
    end
    data_file_path = nothing
    if save
        data_file_path = generate_save_file_path("h5", experiment_name, save_path)
        h5open(data_file_path, "w") do data_file
            write(data_file, "dynamics_type", Integer(dynamics_type))
            write(data_file, "gate_count", gate_count)
            write(data_file, "gate_time", gate_time)
            write(data_file, "gate_type", Integer(gate_type))
            write(data_file, "save_file_path", isnothing(save_file_path) ? "" : save_file_path)
            write(data_file, "seed", seed)
            write(data_file, "states", states)
            write(data_file, "fidelities", fidelities)
            write(data_file, "run_time", string(run_time))
            write(data_file, "dt", dt)
            write(data_file, "negi_h0", Array(negi_h0))
        end
        println("Saved simulation to $(data_file_path)")
    end
    return data_file_path
end


"""
this method is for testing purposes, not meant to be nice
"""
function test_integration(gate_count, gate_type;
    save_file_path=nothing,
    controls_dt_inv=DT_PREF_INV,
    adaptive=DEQJL_ADAPTIVE, dynamics_type=schroed,
    dt=DT_PREF, save=true, save_type=jl, seed=0,
    solver=DifferentialEquations.Vern9, print_final=false,
    negi_h0=NEGI_H0_ISO, rkstep=rk6_step, deqjl=false)
    # grab
    if isnothing(save_file_path)
        controls = Array{Float64, 2}([0 0])
        control_knot_count = 0
        gate_time = DT_GTM[dynamics_type]
    else
        (controls, gate_time) = grab_controls(save_file_path; save_type=save_type)
        control_knot_count = Int(floor(gate_time * controls_dt_inv))
    end
    dt_inv = 1 / dt
    evolution_time = gate_time * gate_count
    knot_count = Int(ceil(evolution_time * dt_inv))
    # save_times = Array(0:1:gate_count) * gate_time
    save_times = Array(1:knot_count) * dt
    
    # integrate
    dynamics = DT_DYN[dynamics_type]
    state_type = DT_ST[dynamics_type]
    if state_type == st_state
        initial_state = gen_rand_state_iso(;seed=seed)
    elseif state_type == st_density
        initial_state =  gen_rand_density_iso(;seed=seed)
    end
    tspan = (0., evolution_time)
    params = SimParams(controls, control_knot_count, controls_dt_inv,
                       negi_h0, dt_inv)
    if deqjl
        prob = ODEProblem(dynamics, initial_state, tspan, params)
        result = solve(prob, solver(), dt=dt, saveat=save_times,
                       maxiters=DEQJL_MAXITERS, adaptive=adaptive)
        states = result.u
        final_state = result.u[end]
    else
        time = 0.
        state = initial_state
        if state_type == st_state
            states = zeros(knot_count, STATE_SIZE_ISO)
        elseif state_type == st_density
            states = zeros(knot_count, STATE_SIZE_ISO, STATE_SIZE_ISO)
        end
        for i = 1:knot_count
            time = time + dt
            state = rkstep(dynamics, state, params, time, dt)
            if state_type == st_state
                states[i, :] = state
            else
                states[i, :, :] = state
            end
        end
        final_state = state
    end

    # obtain fidelity
    gate = GT_GATE[gate_type]

    if state_type == st_state
        target_state = gate * initial_state
        fidelity = abs(fidelity_vec_iso(final_state, target_state))
    elseif state_type == st_density
        target_state = gate * initial_state * gate'
        fidelity = abs(fidelity_mat_iso(final_state, target_state))
    end
    fidelities = [1., fidelity]

    # obtain norms
    if deqjl
        norms = [states[i]'states[i] for i = 1:size(states)[1]]
    else
        norms = [states[i, :]'states[i, :] for i = 1:size(states)[1]]
    end
    min_norm = minimum(norms)
    max_norm = maximum(norms)
    avg_norm = mean(norms)
    println("min_n: $(min_norm), max_norm: $(max_norm), avg_norm: $(avg_norm), f: $(fidelities[end])")
    return
end


# RK6 due to Butcher
# https://github.com/SciML/DiffEqDevTools.jl/blob/master/src/ode_tableaus.jl#L1261
function rk6_step(dynamics, x, u::SimParams, t::Real, dt::Real)
    a21 = 1//2-1//10*5^(1//2)
    a31 = -1//10*5^(1//2)
    a32 = 1//2+1//5*5^(1//2)
    a41 = 7//20*5^(1//2)-3//4
    a42 = 1//4*5^(1//2)-1//4
    a43 = 3//2-7//10*5^(1//2)
    a51 = 1//12-1//60*5^(1//2)
    a53 = 1//6
    a54 = 7//60*5^(1//2)+1//4
    a61 = 1//60*5^(1//2)+1//12
    a63 = 3//4-5//12*5^(1//2)
    a64 = 1//6
    a65 = -1//2+3//10*5^(1//2)
    a71 = 1//6
    a73 = -55//12+25//12*5^(1//2)
    a74 = -7//12*5^(1//2)-25//12
    a75 = 5-2*5^(1//2)
    a76 = 5//2+1//2*5^(1//2)
    b1 = 1//12
    b5 = 5//12
    b6 = 5//12
    b7 = 1//12
    c2 = 1//2-1//10*5^(1//2)
    c3 = 1//2+1//10*5^(1//2)
    c4 = 1//2-1//10*5^(1//2)
    c5 = 1//2+1//10*5^(1//2)
    c6 = 1//2-1//10*5^(1//2)
    c7 = 1

    k1 = dynamics(x, u, t) * dt
    k2 = dynamics(x + a21 * k1, u, t + dt * c2) * dt
    k3 = dynamics(x + a31 * k1 + a32 * k2, u, t + dt * c3) * dt
    k4 = dynamics(x + a41 * k1 + a42 * k2 + a43 * k3, u, t + dt * c4) * dt
    k5 = dynamics(x + a51 * k1            + a53 * k3 + a54 * k4, u, t + dt * c5) * dt
    k6 = dynamics(x + a61 * k1            + a63 * k3 + a64 * k4 + a65 * k5, u, t + dt * c6) * dt
    k7 = dynamics(x + a71 * k1            + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6, u, t + dt * c7) * dt
    return x + (b1 * k1 + b5 * k5 + b6 * k6 + b7 * k7)
end


function rk4_step(dynamics, x, u::SimParams, t::Real, dt::Real)
	k1 = dynamics(x,        u, t       )*dt
	k2 = dynamics(x + k1/2, u, t + dt/2)*dt
	k3 = dynamics(x + k2/2, u, t + dt/2)*dt
	k4 = dynamics(x + k3,   u, t + dt  )*dt
	return x + (k1 + 2k2 + 2k3 + k4)/6
end


function rk3_step(dynamics, x, u::SimParams, t::Real, dt::Real)
    k1 = dynamics(x,             u, t       )*dt;
    k2 = dynamics(x + k1/2,      u, t + dt/2)*dt;
    k3 = dynamics(x - k1 + 2*k2, u, t + dt  )*dt;
    return x + (k1 + 4*k2 + k3)/6
end

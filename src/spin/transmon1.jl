"""
transmon1.jl - vanilla
"""

# paths
WDIR = abspath(@__DIR__, "../../")
const EXPERIMENT_META = "spin"
include(joinpath(WDIR, "src", EXPERIMENT_META, EXPERIMENT_META * ".jl"))
const EXPERIMENT_NAME = "transmon1"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

using Altro
using HDF5
using LinearAlgebra
using StaticArrays

# redefine constants
# TODO: SAN UPDATE THIS
const ω_q = 2π * 3.9 #GHz
const A_MAX = 2π * 0.3 #GHz
const NEGI_H0_ISO = get_mat_iso(-1im * ω_q * SIGMAZ / 2)

# model
struct Model{TH,Tis,Tic} <: AbstractModel
    # problem size
    n::Int
    m::Int
    control_count::Int
    # problem
    Hs::Vector{TH}
    # indices
    state1_idx::Tis
    state2_idx::Tis
    controls_idx::Tic
    dcontrols_idx::Tic
    d2controls_idx::Tic
end

function Model(M_, Md_, V_, Hs)
    # problem size
    control_count = 1
    state_count = HDIM
    n = state_count * HDIM_ISO + 2 * control_count
    m = control_count
    # state indices
    state1_idx = V(1:HDIM_ISO)
    state2_idx = V(state1_idx[end] + 1:state1_idx[end] + HDIM_ISO)
    controls_idx = V(state2_idx[end] + 1:state2_idx[end] + control_count)
    dcontrols_idx = V(controls_idx[end] + 1:controls_idx[end] + control_count)
    # control indices
    d2controls_idx = V(1:control_count)
    # types
    TH = typeof(Hs[1])
    Tis = typeof(state1_idx)
    Tic = typeof(controls_idx)
    return Model{TH,Tis,Tic}(n, m, control_count, Hs, state1_idx, state2_idx,
                             controls_idx, dcontrols_idx, d2controls_idx)
end

@inline Base.size(model::Model) = model.n, model.m
# vector and matrix constructors (use CPU arrays)
@inline M(mat_) = mat_
@inline Md(mat_) = mat_
@inline V(vec_) = vec_

# dynamics
abstract type EXP <: Explicit end

function Altro.discrete_dynamics(::Type{EXP}, model::Model,
                              astate::AbstractVector,
                              acontrol::AbstractVector, time::Real, dt::Real)
    # get hamiltonian and unitary
    H = dt * (
        model.Hs[1]
        + astate[model.controls_idx[1]] * model.Hs[2]
    )
    U = exp(H)
    # propagate state
    state1 = U * astate[model.state1_idx]
    state2 = U * astate[model.state2_idx]
    # propagate controls
    controls = astate[model.dcontrols_idx] .* dt + astate[model.controls_idx]
    # propagate dcontrols
    dcontrols = acontrol[model.d2controls_idx] .* dt + astate[model.dcontrols_idx]
    # construct astate
    astate_ = [state1; state2; controls; dcontrols]
    return astate_
end

# main
function run_traj(;gate_type=xpi, evolution_time=4., dt=1e-2, verbose=true,
                  smoke_test=false, save=true, benchmark=false,
                  pn_steps=2, max_penalty=1e11,
                  max_iterations=Int64(2e5),
                  max_cost=1e8, ilqr_ctol=1e-2, ilqr_gtol=1e-4,
                  ilqr_max_iterations=300, penalty_scaling=10., max_state_value=1e10,
                  max_control_value=1e10, qs=[1e0, 1e0, 1e0, 5e-1])
    # model configuration
    Hs = [M(H) for H in (NEGI_H0_ISO, NEGI_H1_ISO)]
    model = Model(M, Md, V, Hs)
    n, m = size(model)
    N = Int(floor(evolution_time / dt)) + 1
    t0 = 0.

    # initial state
    x0 = zeros(n)
    x0[model.state1_idx] = IS1_ISO
    x0[model.state2_idx] = IS2_ISO
    x0 = V(x0)

    # final state
    if gate_type == xpi
        target_state1 = IS2_ISO
        target_state2 = IS1_ISO
    elseif gate_type == xpiby2
        target_state1 = XPIBY21_ISO
        target_state2 = XPIBY22_ISO
    end
    xf = zeros(n)
    xf[model.state1_idx] = target_state1
    xf[model.state2_idx] = target_state2
    xf = V(xf)

    # bound constraints
    x_max_amid = fill(Inf, n)
    x_max_abnd = fill(Inf, n)
    x_min_amid = fill(-Inf, n)
    x_min_abnd = fill(-Inf, n)
    u_max_amid = fill(Inf, m)
    u_max_abnd = fill(Inf, m)
    u_min_amid = fill(-Inf, m)
    u_min_abnd = fill(-Inf, m)
    # constrain the control amplitudes
    x_max_amid[model.controls_idx] .= A_MAX
    x_min_amid[model.controls_idx] .= -A_MAX
    # control amplitudes go to zero at boundary
    x_max_abnd[model.controls_idx] .= 0
    x_min_abnd[model.controls_idx] .= 0
    # vectorize
    x_max_amid = V(x_max_amid)
    x_max_abnd = V(x_max_abnd)
    x_min_amid = V(x_min_amid)
    x_min_abnd = V(x_min_abnd)
    u_max_amid = V(u_max_amid)
    u_max_abnd = V(u_max_abnd)
    u_min_amid = V(u_min_amid)
    u_min_abnd = V(u_min_abnd)

    # constraints
    constraints = ConstraintList(n, m, N, M, V)
    bc_amid = BoundConstraint(n, m, x_max_amid, x_min_amid, u_max_amid, u_min_amid, M, V)
    # add_constraint!(constraints, bc_amid, V(2:N-2))
    bc_abnd = BoundConstraint(n, m, x_max_abnd, x_min_abnd, u_max_abnd, u_min_abnd, M, V)
    # add_constraint!(constraints, bc_abnd, V(N-1:N-1))
    goal_idxs = V([model.state1_idx; model.state2_idx])
    gc_f = GoalConstraint(n, m, xf, goal_idxs, M, V)
    add_constraint!(constraints, gc_f, V(N:N))

    # initial trajectory
    X0 = [V(zeros(n)) for k = 1:N]
    X0[1] .= x0
    U0 = [V([
        fill(1e-4, model.control_count);
    ]) for k = 1:N-1]
    ts = V(zeros(N))
    ts[1] = t0
    for k = 1:N-1
        ts[k + 1] = ts[k] + dt
        discrete_dynamics!(X0[k + 1], EXP, model, X0[k], U0[k], ts[k], dt)
    end

    # cost function
    Q = V(zeros(n))
    Q[model.state1_idx] .= qs[1]
    Q[model.state2_idx] .= qs[1]
    Q[model.controls_idx] .= qs[2]
    Q[model.dcontrols_idx] .= qs[3]
    Q = Diagonal(Q)
    Qf = Q * N
    R = V(zeros(m))
    R[model.d2controls_idx] .= qs[4]
    R = Diagonal(R)
    objective = LQRObjective(Q, Qf, R, xf, n, m, N, M, V)

    # build problem
    prob = Problem(EXP, model, objective, constraints, X0, U0, ts, N, M, Md, V)
    # options
    verbose_pn = verbose ? true : false
    verbose_ = verbose ? 2 : 0
    ilqr_max_iterations = smoke_test ? 1 : ilqr_max_iterations
    al_max_iterations = smoke_test ? 1 : 30
    n_steps = smoke_test ? 1 : pn_steps
    opts = SolverOptions(
        penalty_max=max_penalty, verbose_pn=verbose_pn, verbose=verbose_,
        projected_newton=true, ilqr_max_iterations=ilqr_max_iterations,
        al_max_iterations=al_max_iterations,
        iterations=max_iterations,
        max_cost_value=max_cost, ilqr_ctol=ilqr_ctol, ilqr_gtol=ilqr_gtol,
        penalty_scaling=penalty_scaling, max_state_value=max_state_value,
        max_control_value=max_control_value
    )

    # solve
    solver = ALTROSolver(prob, opts)
    if benchmark
        benchmark_result = Altro.benchmark_solve!(solver)
    else
        benchmark_result = nothing
        Altro.solve!(solver)
    end
    println("status: $(solver.stats.status)")

    # post-process
    acontrols_raw = Altro.controls(solver)
    acontrols_arr = permutedims(reduce(hcat, map(Array, acontrols_raw)), [2, 1])
    astates_raw = Altro.states(solver)
    astates_arr = permutedims(reduce(hcat, map(Array, astates_raw)), [2, 1])
    state1_idx_arr = Array(model.state1_idx)
    controls_idx_arr = Array(model.controls_idx)
    dcontrols_idx_arr = Array(model.dcontrols_idx)
    d2controls_idx_arr = Array(model.d2controls_idx)
    max_v, max_v_info = Altro.max_violation_info(solver)
    iterations_ = Altro.iterations(solver)

    result = Dict(
        "acontrols" => acontrols_arr,
        "astates" => astates_arr,
        "dt" => dt,
        "ts" => ts,
        "state1_idx" => state1_idx_arr,
        "controls_idx" => controls_idx_arr,
        "dcontrols_idx" => dcontrols_idx_arr,
        "d2controls_idx" => d2controls_idx_arr,
        "evolution_time" => evolution_time,
        "max_v" => max_v,
        "max_v_info" => max_v_info,
        "qs" => qs,
        "iterations" => iterations_,
        "hdim_iso" => HDIM_ISO,
        "save_type" => Int(jl),
        "max_penalty" => max_penalty,
        "ilqr_ctol" => ilqr_ctol,
        "ilqr_gtol" => ilqr_gtol,
        "iterations" => iterations_,
        "max_iterations" => max_iterations,
        "pn_steps" => pn_steps,
        "max_cost" => max_cost,
    )
    
    # save
    if save
        save_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
        println("Saving this optimization to $(save_file_path)")
        h5open(save_file_path, "cw") do save_file
            for key in keys(result)
                write(save_file, key, result[key])
            end
        end
        result["save_file_path"] = save_file_path
    end

    result = benchmark ? benchmark_result : result

    return result
end

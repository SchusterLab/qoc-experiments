"""
spin1.jl
"""

WDIR = joinpath(@__DIR__, "../..")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

using Altro
using ForwardDiff
using HDF5
using LinearAlgebra
using RobotDynamics
using TrajectoryOptimization
const RD = RobotDynamics
const TO = TrajectoryOptimization

# paths
const EXPERIMENT_META = "spin"
const EXPERIMENT_NAME = "spin1"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

# model
struct Model{TH,Tis,Tic,Tid} <: AbstractModel
    # problem size
    n::Int
    m::Int
    # problem
    Hs::Vector{TH}
    time_optimal::Bool
    # indices
    state1_idx::Tis
    state2_idx::Tis
    controls_idx::Tic
    dcontrols_idx::Tic
    d2controls_idx::Tic
    dt_idx::Tid
end

function Model(M_, Md_, V_, Hs, time_optimal)
    # problem size
    control_count = 1
    state_count = HDIM
    n = state_count * HDIM_ISO + 2 * control_count
    m = time_optimal ? control_count + 1 : control_count
    # state indices
    state1_idx = V(1:HDIM_ISO)
    state2_idx = V(state1_idx[end] + 1:state1_idx[end] + HDIM_ISO)
    controls_idx = V(state2_idx[end] + 1:state2_idx[end] + control_count)
    dcontrols_idx = V(controls_idx[end] + 1:controls_idx[end] + control_count)
    # control indices
    d2controls_idx = V(1:control_count)
    dt_idx = V(d2controls_idx[end] + 1:d2controls_idx[end] + 1)
    # types
    TH = typeof(Hs[1])
    Tis = typeof(state1_idx)
    Tic = typeof(controls_idx)
    Tid = typeof(dt_idx)
    return Model{TH,Tis,Tic,Tid}(n, m, Hs, time_optimal, state1_idx, state2_idx,
                                 controls_idx, dcontrols_idx, d2controls_idx, dt_idx)
end

@inline Base.size(model::Model) = model.n, model.m
# vector and matrix constructors (use CPU arrays)
@inline M(mat_) = mat_
@inline Md(mat_) = mat_
@inline V(vec_) = vec_

# dynamics
abstract type EXP <: RD.Explicit end

function RD.discrete_dynamics(::Type{EXP}, model::Model,
                              astate::AbstractVector,
                              acontrol::AbstractVector, time::Real, dt_::Real)
    dt = !model.time_optimal ? dt_ : acontrol[model.dt_idx[1]]^2
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
    controls = astate[model.dcontrols_idx[1]] .* dt + astate[model.controls_idx[1]]
    # propagate dcontrols
    dcontrols = acontrol[model.d2controls_idx[1]] .* dt + astate[model.dcontrols_idx[1]]
    # construct astate
    astate_ = [state1; state2; controls; dcontrols]
    return astate_
end

function run_traj(;evolution_time=20., dt=DT_PREF, verbose=true,
                  time_optimal=false, qs=[1e0, 1e-1, 1e-1, 1e-1, 1e-1], smoke_test=false,
                  save=true, benchmark=false, max_iterations=10000, bp_reg_fp=10.,
                  dJ_counter_limit=20, bp_reg_type=:control, projected_newton=true)
    # initialize model
    Hs = [M(H) for H in (NEGI_H0_ISO, NEGI_H1_ISO)]
    model = Model(M, Md, V, Hs, time_optimal)
    n, m = size(model)
    t0 = 0.
    dt_max = 2 * dt
    dt_min = dt / 1e1
    sqrt_dt_max = sqrt(dt_max)
    sqrt_dt_min = sqrt(dt_min)

    # initial state
    x0 = zeros(n)
    x0[model.state1_idx] = IS1_ISO
    x0[model.state2_idx] = IS2_ISO
    x0 = V(x0)

    # target state
    xf = zeros(n)
    xf[model.state1_idx] = ZPIBY21_ISO
    xf[model.state2_idx] = ZPIBY22_ISO
    xf = V(xf)

    # bound constraints
    x_max = fill(Inf, n)
    x_max_boundary = fill(Inf, n)
    x_min = fill(-Inf, n)
    x_min_boundary = fill(-Inf, n)
    u_max = fill(Inf, m)
    u_max_boundary = fill(Inf, m)
    u_min = fill(-Inf, m)
    u_min_boundary = fill(-Inf, m)
    # constrain the control amplitudes
    x_max[model.controls_idx[1]] = A_MAX
    x_min[model.controls_idx[1]] = -A_MAX
    # control amplitudes go to zero at boundary
    x_max_boundary[model.controls_idx] .= 0
    x_min_boundary[model.controls_idx] .= 0
    # constraint dt
    if time_optimal
        u_max[model.dt_idx] .= sqrt_dt_max
        u_max_boundary[model.dt_idx] .= sqrt_dt_max
        u_min[model.dt_idx] .= sqrt_dt_min
        u_min_boundary[model.dt_idx] .= sqrt_dt_min
    end
    # vectorize
    x_max = V(x_max)
    x_max_boundary = V(x_max_boundary)
    x_min = V(x_min)
    x_min_boundary = V(x_min_boundary)
    u_max = V(u_max)
    u_max_boundary = V(u_max_boundary)
    u_min = V(u_min)
    u_min_boundary = V(u_min_boundary)
    
    # initial trajectory
    N = Int(floor(evolution_time / dt)) + 1
    X0 = [V(zeros(n)) for k = 1:N]
    X0[1] .= x0
    U0 = [V([
        fill(1e-4, 1);
        fill(dt, time_optimal ? 1 : 0);
    ]) for k = 1:N-1]
    ts = V(zeros(N))
    ts[1] = t0
    for k = 1:N-1
        ts[k + 1] = ts[k] + dt
        RD.discrete_dynamics!(X0[k + 1], EXP, model, X0[k], U0[k], ts[k], dt)
    end
    
    # cost function
    Q = zeros(n)
    Q[model.state1_idx] .= qs[1]
    Q[model.state2_idx] .= qs[1]
    Q[model.controls_idx] .= qs[2]
    Q[model.dcontrols_idx] .= qs[3]
    Q = Diagonal(V(Q))
    # Qf = Q .* N
    Qf = Q
    R = zeros(m)
    R[model.d2controls_idx] .= qs[4]
    if time_optimal
        R[model.dt_idx] .= qs[5]
    end
    R = Diagonal(V(R))
    objective = LQRObjective(Q, Qf, R, xf, n, m, N, M, V)

    # create constraints
    control_amp = TO.BoundConstraint(n, m, x_max, x_min, u_max, u_min, M, V)
    control_amp_boundary = TO.BoundConstraint(n, m, x_max_boundary, x_min_boundary,
                                              u_max_boundary, u_min_boundary, M, V)
    target_astate_constraint = TO.GoalConstraint(n, m, xf, V([model.state1_idx; model.state2_idx]),
                                                 M, V)
    # add constraints
    constraints = TO.ConstraintList(n, m, N, M, V)
    TO.add_constraint!(constraints, control_amp, V(2:N-2))
    TO.add_constraint!(constraints, control_amp_boundary, V(N-1:N-1))
    TO.add_constraint!(constraints, target_astate_constraint, V(N:N))
    
    # build problem
    prob = Problem(EXP, model, objective, constraints, X0, U0, ts, N, M, Md, V)
    # options
    verbose_pn = verbose ? true : false
    verbose_ = verbose ? 2 : 0
    ilqr_max_iterations = smoke_test ? 1 : 300
    al_max_iterations = smoke_test ? 1 : 30
    n_steps = smoke_test ? 1 : 2
    opts = SolverOptions(
        verbose_pn=verbose_pn, verbose=verbose_,
        ilqr_max_iterations=ilqr_max_iterations,
        al_max_iterations=al_max_iterations, n_steps=n_steps, iterations=max_iterations,
        bp_reg_fp=bp_reg_fp, dJ_counter_limit=dJ_counter_limit, bp_reg_type=bp_reg_type,
        projected_newton=false,
    )
    # solve
    solver = ALTROSolver(prob, opts)
    if benchmark
        benchmark_result = Altro.benchmark_solve!(solver)
    else
        benchmark_result = nothing
        Altro.solve!(solver)
    end
    if verbose
        println("status: $(solver.stats.status)")
    end
    
    # post-process
    acontrols_raw = Altro.controls(solver)
    acontrols_arr = permutedims(reduce(hcat, map(Array, acontrols_raw)), [2, 1])
    astates_raw = Altro.states(solver)
    astates_arr = permutedims(reduce(hcat, map(Array, astates_raw)), [2, 1])
    state1_idx_arr = Array(model.state1_idx)
    state2_idx_arr = Array(model.state2_idx)
    controls_idx_arr = Array(model.controls_idx)
    dcontrols_idx_arr = Array(model.dcontrols_idx)
    d2controls_idx_arr = Array(model.d2controls_idx)
    dt_idx_arr = Array(model.dt_idx)
    max_v, max_v_info = Altro.max_violation_info(solver)
    iterations_ = Altro.iterations(solver)
    if time_optimal
        ts = cumsum(map(x -> x^2, acontrols_arr[:,model.dt_idx[1]]))
    end

    result = Dict(
        "acontrols" => acontrols_arr,
        "astates" => astates_arr,
        "dt" => dt,
        "ts" => ts,
        "state1_idx" => state1_idx_arr,
        "state2_idx" => state2_idx_arr,
        "controls_idx" => controls_idx_arr,
        "dcontrols_idx" => dcontrols_idx_arr,
        "d2controls_idx" => d2controls_idx_arr,
        "dt_idx" => dt_idx_arr,
        "evolution_time" => evolution_time,
        "max_v" => max_v,
        "max_v_info" => max_v_info,
        "qs" => qs,
        "iterations" => iterations_,
        "time_optimal" => Integer(time_optimal),
        "hdim_iso" => HDIM_ISO,
        "save_type" => Int(jl)
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

function to_gif()
    xlims = (0., 105.)
    ylims = (-0.5, 0.5)
    evolution_time = 100.
    qs = [1e0, 1e2, 1e2, 1e2, 1e-1]
    dpi = 300
    anim = Plots.@animate for i = 0:10
        if i == 0
            N = 1000
            controls = zeros(N + 1)
            ts = zeros(N + 1)
            for i = 2:N + 1
                ts[i] = ts[i - 1] + 1e-1
                controls[i] = controls[i - 1] + 1e-5
            end
        else
            res = run_traj(evolution_time=evolution_time, qs=qs, time_optimal=true,
                           projected_newton=false, verbose=false, max_iterations=i, save=false)
            controls = res["astates"][1:end-1, res["controls_idx"][1]]
            controls ./= 2π
            ts = res["ts"]
            insert!(ts, 1, 0.)
            insert!(controls, 1, 0.)
        end
        Plots.plot(ts, controls, xlims=xlims, ylims=ylims, label=nothing, dpi=dpi, linewidth=10)
        # Plots.scatter!(ts, controls, markersize=2, label=nothing)
        Plots.xlabel!("t (ns)")
        Plots.ylabel!("a (GHz)")
    end
    plot_file_path = generate_file_path("gif", EXPERIMENT_NAME, SAVE_PATH)
    Plots.gif(anim, plot_file_path, fps=1)
end

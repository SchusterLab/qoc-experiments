"""
mm1s.jl - first multimode experiment w/ sparsity
"""

WDIR = joinpath(@__DIR__, "../..")
include(joinpath(WDIR, "src", "mm", "mm.jl"))

using Altro
using HDF5
using LinearAlgebra
using RobotDynamics
using SparseArrays
using StaticArrays
using TrajectoryOptimization
const RD = RobotDynamics
const TO = TrajectoryOptimization

# paths
const EXPERIMENT_META = "mm"
const EXPERIMENT_NAME = "mm1"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

# problem
const CONTROL_COUNT = 4
const STATE_COUNT = 1
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 2 * CONTROL_COUNT
const ACONTROL_SIZE = CONTROL_COUNT #+ 1
# state indices
const STATE1_IDX = 1:HDIM_ISO
const CONTROLS_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT
const DT_IDX = D2CONTROLS_IDX[end] + 1:D2CONTROLS_IDX[end] + 1

# model
struct Model <: AbstractModel
end
@inline RD.state_dim(::Model) = ASTATE_SIZE_BASE
@inline RD.control_dim(::Model) = ACONTROL_SIZE

# dynamics
abstract type EXP <: RD.Explicit end

const NEGI_H0ROT_ISO_ = sparse(NEGI_H0ROT_ISO)
const NEGI_H1R_ISO_ = sparse(NEGI_H1R_ISO)
const NEGI_H1I_ISO_ = sparse(NEGI_H1I_ISO)
const NEGI_H2R_ISO_ = sparse(NEGI_H2R_ISO)
const NEGI_H2I_ISO_ = sparse(NEGI_H2I_ISO)

function RD.discrete_dynamics(::Type{EXP}, model::Model, astate::AbstractVector,
                              acontrol::AbstractVector, time::Real, dt::Real)
    negi_h = (
        NEGI_H0ROT_ISO_
        + astate[CONTROLS_IDX[1]] * NEGI_H1R_ISO_
        + astate[CONTROLS_IDX[2]] * NEGI_H1I_ISO_
        + astate[CONTROLS_IDX[3]] * NEGI_H2R_ISO_
        + astate[CONTROLS_IDX[4]] * NEGI_H2I_ISO_
    )
    h_prop = exp_(negi_h * dt)
    state1 =  h_prop * astate[STATE1_IDX]
    controls = astate[CONTROLS_IDX] + astate[DCONTROLS_IDX] * dt
    dcontrols = astate[DCONTROLS_IDX] + acontrol[D2CONTROLS_IDX] * dt

    astate_ = [
        state1; controls; dcontrols;
    ]

    return astate_
end

function run_traj(;fock_state=0, evolution_time=800., dt_inv=1., verbose=true,
                  sqrtbp=false, derivative_order=0,
                  qs=[1e0, 1e-1, 1e-1, 1e-1],
                  smoke_test=false, constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e11, save=true, max_iterations=Int64(2e5),
                  max_cost_value=1e8, benchmark=false, static_bp=false)
    model = Model()
    n_ = state_dim(model)
    m_ = control_dim(model)
    t0 = 0.

    # initial state
    x0 = zeros(n_)
    x0[STATE1_IDX] = IS1_ISO

    # target state
    xf = zeros(n_)
    cavity_state = zeros(CAVITY_STATE_COUNT)
    cavity_state[fock_state + 1] = 1
    xf[STATE1_IDX] = get_vec_iso(kron(cavity_state, TRANSMON_G))

    # control amplitude constraint at boundary
    x_max = fill(Inf, n_)
    x_max[CONTROLS_IDX[1:2]] .= MAX_AMP_NORM_TRANSMON
    x_max[CONTROLS_IDX[3:4]] .= MAX_AMP_NORM_CAVITY
    u_max = fill(Inf, m_)
    x_min = fill(-Inf, n_)
    x_min[CONTROLS_IDX[1:2]] .= -MAX_AMP_NORM_TRANSMON
    x_min[CONTROLS_IDX[3:4]] .= -MAX_AMP_NORM_CAVITY
    u_min = fill(-Inf, m_)
    # control amplitude constraint at boundary
    x_max_boundary = fill(Inf, n_)
    x_max_boundary[CONTROLS_IDX] .= 0
    u_max_boundary = fill(Inf, m_)
    x_min_boundary = fill(-Inf, n_)
    x_min_boundary[CONTROLS_IDX] .= 0
    u_min_boundary = fill(-Inf, m_)

    # initial trajectory
    dt = dt_inv^(-1)
    N_ = Int(floor(evolution_time * dt_inv)) + 1
    U0 = [[
        fill(1e-6, 2);
        fill(1e-6, 2);
    ] for k = 1:N_-1]
    X0 = [[
        fill(NaN, n_);
    ] for k = 1:N_]
    Z = Traj(X0, U0, dt * ones(N_))
    
    # cost function
    Q = zeros(n_)
    Q[STATE1_IDX] .= qs[1]
    Q[CONTROLS_IDX] .= qs[2]
    Q[DCONTROLS_IDX] .= qs[3]
    # Q = Diagonal(SVector{n_}(Q))
    Q = Diagonal(Q)
    Qf = Q * N_
    R = zeros(m_)
    R[D2CONTROLS_IDX] .= qs[4]
    # R = Diagonal(SVector{m_}(R))
    R = Diagonal(R)
    objective = LQRObjective(Q, R, Qf, xf, N_)

    # must satisfy control amplitude constraints
    control_amp = BoundConstraint(n_, m_; x_max=x_max, x_min=x_min, u_max=u_max, u_min=u_min)
    # must statisfy controls start and stop at 0
    control_amp_boundary = BoundConstraint(n_, m_; x_max=x_max_boundary, x_min=x_min_boundary,
                                           u_max=u_max_boundary, u_min=u_min_boundary)
    # must reach target state, must have integral of controls = 0
    target_astate_constraint = GoalConstraint(xf, STATE1_IDX)
    
    constraints = TO.ConstraintList(n_, m_, N_)
    add_constraint!(constraints, control_amp, 2:N_-2)
    add_constraint!(constraints, control_amp_boundary, 1:1)
    add_constraint!(constraints, control_amp_boundary, N_-1:N_-1)
    add_constraint!(constraints, target_astate_constraint, N_:N_)
    
    prob = Problem{EXP}(model, objective, constraints, x0, xf, Z, N_, t0, evolution_time)
    solver = ALTROSolver(prob)
    verbose_pn = verbose ? true : false
    verbose_ = verbose ? 2 : 0
    iterations_inner = smoke_test ? 1 : 300
    iterations_outer = smoke_test ? 1 : 30
    n_steps = smoke_test ? 1 : pn_steps
    set_options!(solver, square_root=sqrtbp, constraint_tolerance=constraint_tol,
                 projected_newton_tolerance=al_tol, n_steps=n_steps,
                 penalty_max=max_penalty, verbose_pn=verbose_pn, verbose=verbose_,
                 projected_newton=true, iterations_inner=iterations_inner,
                 iterations_outer=iterations_outer, iterations=max_iterations,
                 max_cost_value=max_cost_value, static_bp=static_bp)
    if benchmark
        benchmark_result = Altro.benchmark_solve!(solver)
    else
        benchmark_result = nothing
        Altro.solve!(solver)
    end

    # post-process
    acontrols_raw = TO.controls(solver)
    acontrols_arr = permutedims(reduce(hcat, map(Array, acontrols_raw)), [2, 1])
    astates_raw = TO.states(solver)
    astates_arr = permutedims(reduce(hcat, map(Array, astates_raw)), [2, 1])
    Q_raw = Array(Q)
    Q_arr = [Q_raw[i, i] for i in 1:size(Q_raw)[1]]
    Qf_raw = Array(Qf)
    Qf_arr = [Qf_raw[i, i] for i in 1:size(Qf_raw)[1]]
    R_raw = Array(R)
    R_arr = [R_raw[i, i] for i in 1:size(R_raw)[1]]
    cidx_arr = Array(CONTROLS_IDX)
    d2cidx_arr = Array(D2CONTROLS_IDX)
    cmax = TrajectoryOptimization.max_violation(solver)
    cmax_info = TrajectoryOptimization.findmax_violation(TO.get_constraints(solver))
    iterations_ = Altro.iterations(solver)

    result = Dict(
        "acontrols" => acontrols_arr,
        "controls_idx" => cidx_arr,
        "d2controls_dt2_idx" => d2cidx_arr,
        "evolution_time" => evolution_time,
        "astates" => astates_arr,
        "Q" => Q_arr,
        "Qf" => Qf_arr,
        "R" => R_arr,
        "cmax" => cmax,
        "cmax_info" => cmax_info,
        "dt" => dt,
        "derivative_order" => derivative_order,
        "sqrtbp" => Integer(sqrtbp),
        "max_penalty" => max_penalty,
        "constraint_tol" => constraint_tol,
        "al_tol" => al_tol,
        "save_type" => Integer(jl),
        "iterations" => iterations_,
        "max_iterations" => max_iterations,
        "pn_steps" => pn_steps,
        "max_cost_value" => max_cost_value,
        "static_bp" => Integer(static_bp),
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

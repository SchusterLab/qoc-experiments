"""
spinspin0.jl
"""

using Altro
using HDF5
using LinearAlgebra
using RobotDynamics
using StaticArrays
using TrajectoryOptimization

WDIR = get(ENV, "QOC_EXPERIMENTS_PATH", "../../")
include(joinpath(WDIR, "src", "vslq", "vslq.jl"))

# paths
const EXPERIMENT_META = "vslq"
const EXPERIMENT_NAME = "spinspin0"
const SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# problem
const CONTROL_COUNT = 2
const STATE_COUNT = 4
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 2 * CONTROL_COUNT
const INITIAL_STATE1 = II_ISO_1
const INITIAL_STATE2 = II_ISO_2
const INITIAL_STATE3 = II_ISO_3
const INITIAL_STATE4 = II_ISO_4
# astate indices
const STATE1_IDX = 1:HDIM_ISO
const STATE2_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_ISO
const STATE3_IDX = STATE2_IDX[end] + 1:STATE2_IDX[end] + HDIM_ISO
const STATE4_IDX = STATE3_IDX[end] + 1:STATE3_IDX[end] + HDIM_ISO
const CONTROLS_IDX = STATE4_IDX[end] + 1:STATE4_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
# acontrol indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT
const DT_IDX = D2CONTROLS_IDX[end] + 1:D2CONTROLS_IDX[end] + 1

struct Model{TO} <: AbstractModel
    Model(TO::Bool=false) = new{TO}()
end
RobotDynamics.state_dim(::Model{TO}) where TO = ASTATE_SIZE_BASE
RobotDynamics.control_dim(::Model{false}) = CONTROL_COUNT
RobotDynamics.control_dim(::Model{true}) = CONTROL_COUNT + 1


function RobotDynamics.discrete_dynamics(::Type{RK3}, model::Model{TO}, astate::StaticVector,
                                         acontrols::StaticVector, time::Real, dt::Real) where {TO}
    if TO
        dt = acontrols[DT_IDX][1]^2
    end
    negi_h = (
        NEGI_H0_ISO
        + astate[CONTROLS_IDX][1] * NEGI_H1_ISO
        + astate[CONTROLS_IDX][2] * NEGI_H2_ISO
    )
    negi_h_prop = exp(dt * negi_h)
    state1 = negi_h_prop * astate[STATE1_IDX]
    # state2 = negi_h_prop * astate[STATE2_IDX]
    # state3 = negi_h_prop * astate[STATE3_IDX]
    # state4 = negi_h_prop * astate[STATE4_IDX]
    controls = astate[CONTROLS_IDX] + dt * astate[DCONTROLS_IDX]
    dcontrols = astate[DCONTROLS_IDX] + dt * acontrols[D2CONTROLS_IDX]
    astate_ = [
        state1; state2; state3; state4; controls; dcontrols;
    ]
    
    return astate_
end


function run_traj(;gate_type=xpiby2, time_optimal=false,
                  evolution_time=10., sqrtbp=false,
                  integrator_type=rk3, solver_type=altro, smoke_test=false,
                  qs=ones(5), dt_inv=Int64(1e1), save=true, verbose=true,
                  max_penalty=1e11, constraint_tol=1e-8, al_tol=1e-4, pn_steps=2)
    dt = dt_inv^(-1)
    dt_max = (dt_inv / 2)^(-1)
    dt_min = (dt_inv * 2)^(-1)
    N = Int(ceil(evolution_time * dt_inv)) + 1
    model = Model(time_optimal)
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.
    # These initial_state vectors are already wrapped as svectors.
    # We unwrap them, concatenate them with the other astate elements,
    # and wrap the astate as a whole svector.
    x0 = SizedVector{n}([
        Array(INITIAL_STATE1); # ψ1
        Array(INITIAL_STATE2); # ψ2
        Array(INITIAL_STATE3); # ψ3
        Array(INITIAL_STATE4); # ψ4
        zeros(CONTROL_COUNT); # a
        zeros(CONTROL_COUNT); # ∂a
    ])

    # Construct the target state.
    if gate_type == xpiby2
        target_state1 = Array(XPIBY2I_ISO_1)
        target_state2 = Array(XPIBY2I_ISO_2)
        target_state3 = Array(XPIBY2I_ISO_3)
        target_state4 = Array(XPIBY2I_ISO_4)
    end
    xf = SizedVector{n}([
        target_state1;
        target_state2;
        target_state3;
        target_state4;
        zeros(2 * CONTROL_COUNT);
    ])

    # Bound the control amplitude.
    x_max = SizedVector{n}([
        fill(Inf, STATE_COUNT * HDIM_ISO); # ψ1..4
        fill(MAX_AMP, CONTROL_COUNT); # a
        fill(Inf, CONTROL_COUNT); # ∂a
    ])
    x_min = SizedVector{n}([
        fill(-Inf, STATE_COUNT * HDIM_ISO); # ψ1..4
        fill(-MAX_AMP, CONTROL_COUNT); # a
        fill(-Inf, CONTROL_COUNT); # ∂a
    ])
    # Controls start and end at 0.
    x_max_boundary = SizedVector{n}([
        fill(Inf, STATE_COUNT * HDIM_ISO); # ψ1..4
        fill(0, CONTROL_COUNT); # a
        fill(Inf, CONTROL_COUNT); # ∂a
    ])
    x_min_boundary = SizedVector{n}([
        fill(-Inf, STATE_COUNT * HDIM_ISO); #ψ1..4
        fill(0, CONTROL_COUNT); # a
        fill(-Inf, CONTROL_COUNT); # ∂a
    ])
    # Bound dt if time optimal.
    u_max = SizedVector{m}([
        fill(Inf, CONTROL_COUNT); # ∂2a
        fill(sqrt(dt_max), eval(:($time_optimal ? 1 : 0))); # √Δt
    ])
    u_min = SizedVector{m}([
        fill(-Inf, CONTROL_COUNT); # ∂2a
        fill(sqrt(dt_min), eval(:($time_optimal ? 1 : 0))); # √Δt
    ])

    # Generate initial trajectory.
    U0 = [SizedVector{m}([
        fill(1e-5, CONTROL_COUNT); # ∂2a
        fill(sqrt(dt), eval(:($time_optimal ? 1 : 0))); # √Δt
    ]) for k = 1:N - 1]
    X0 = [SizedVector{n}([
        fill(NaN, n);
    ]) for k = 1:N]
    dt_ = time_optimal ? 1 : dt
    Z = Traj(X0, U0, dt_ * ones(N))

    # Define penalties.
    Q = Diagonal(SizedVector{n}([
        fill(qs[1], STATE_COUNT * HDIM_ISO); # ψ1..4
        fill(qs[2], CONTROL_COUNT); # a
        fill(qs[3], CONTROL_COUNT); # ∂a
    ]))
    Qf = Q * N
    R = Diagonal(SizedVector{m}([
        fill(qs[4], CONTROL_COUNT); # ∂2a
        fill(qs[5], eval(:($time_optimal ? 1 : 0))); # √Δt
    ]))
    obj = LQRObjective(Q, R, Qf, xf, N)

    # Must satisfy control amplitude bound.
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # Must statisfy conrols start and stop at 0.
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # Must satisfy dt bound.
    dt_bnd = BoundConstraint(n, m, u_max=u_max, u_min=u_min)
    # Must reach target state.
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX; STATE3_IDX; STATE4_IDX])
    # Must obey unit norm.
    normalization_constraint1 = NormConstraint(n, m, 1, TrajectoryOptimization.Equality(), STATE1_IDX)
    normalization_constraint2 = NormConstraint(n, m, 1, TrajectoryOptimization.Equality(), STATE2_IDX)
    normalization_constraint3 = NormConstraint(n, m, 1, TrajectoryOptimization.Equality(), STATE3_IDX)
    normalization_constraint4 = NormConstraint(n, m, 1, TrajectoryOptimization.Equality(), STATE4_IDX)
    
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N)
    # add_constraint!(constraints, normalization_constraint1, 2:N-1)
    # add_constraint!(constraints, normalization_constraint2, 2:N-1)
    # add_constraint!(constraints, normalization_constraint3, 2:N-1)
    # add_constraint!(constraints, normalization_constraint4, 2:N-1)
    if time_optimal
        add_constraint!(constraints, dt_bnd, 1:N-1)
    end

    # Instantiate problem and solve.
    prob = Problem{IT_RDI[integrator_type]}(model, obj, constraints, x0, xf, Z, N, t0, evolution_time)
    opts = SolverOptions(verbose=verbose)
    if solver_type == alilqr
        solver = AugmentedLagrangianSolver(prob, opts)
        solver.solver_uncon.opts.square_root = sqrtbp
        solver.opts.constraint_tolerance = al_tol
        solver.opts.constraint_tolerance_intermediate = al_tol
        solver.opts.penalty_max = max_penalty
        if smoke_test
            solver.opts.iterations = 1
            solver.solver_uncon.opts.iterations = 1
        end
    elseif solver_type == altro
        solver = ALTROSolver(prob, opts)
        solver.opts.constraint_tolerance = constraint_tol
        solver.solver_al.solver_uncon.opts.square_root = sqrtbp
        solver.solver_al.opts.constraint_tolerance = al_tol
        solver.solver_al.opts.constraint_tolerance_intermediate = al_tol
        solver.solver_al.opts.penalty_max = max_penalty
        solver.solver_pn.opts.constraint_tolerance = constraint_tol
        solver.solver_pn.opts.n_steps = pn_steps
        if smoke_test
            solver.solver_al.opts.iterations = 1
            solver.solver_al.solver_uncon.opts.iterations = 1
            solver.solver_pn.opts.n_steps = 1
        end
    end
    Altro.solve!(solver)

    # Post-process.
    acontrols_raw = controls(solver)
    acontrols_arr = permutedims(reduce(hcat, map(Array, acontrols_raw)), [2, 1])
    astates_raw = TrajectoryOptimization.states(solver)
    astates_arr = permutedims(reduce(hcat, map(Array, astates_raw)), [2, 1])
    Q_raw = Array(Q)
    Q_arr = [Q_raw[i, i] for i in 1:size(Q_raw)[1]]
    Qf_raw = Array(Qf)
    Qf_arr = [Qf_raw[i, i] for i in 1:size(Qf_raw)[1]]
    R_raw = Array(R)
    R_arr = [R_raw[i, i] for i in 1:size(R_raw)[1]]
    cidx_arr = Array(CONTROLS_IDX)
    d2cidx_arr = Array(D2CONTROLS_IDX)
    dtidx_arr = Array(DT_IDX)
    # Square the dts.
    if time_optimal
        acontrols_arr[:, DT_IDX] = acontrols_arr[:, DT_IDX] .^2
    end
    cmax = TrajectoryOptimization.max_violation(solver)
    cmax_info = TrajectoryOptimization.findmax_violation(get_constraints(solver))
    iterations_ = iterations(solver)
    
    # save
    if save
        save_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
        println("Saving this optimization to $(save_file_path)")
        h5open(save_file_path, "cw") do save_file
            write(save_file, "acontrols", acontrols_arr)
            write(save_file, "controls_idx", cidx_arr)
            write(save_file, "d2controls_idx", d2cidx_arr)
            write(save_file, "dt_idx", dtidx_arr)
            write(save_file, "evolution_time", evolution_time)
            write(save_file, "astates", astates_arr)
            write(save_file, "Q", Q_arr)
            write(save_file, "Qf", Qf_arr)
            write(save_file, "R", R_arr)
            write(save_file, "cmax", cmax)
            write(save_file, "cmax_info", cmax_info)
            write(save_file, "sqrtbp", Integer(sqrtbp))
            write(save_file, "max_penalty", max_penalty)
            write(save_file, "ctol", constraint_tol)
            write(save_file, "alko", al_tol)
            write(save_file, "integrator_type", Integer(integrator_type))
            write(save_file, "gate_type", Integer(gate_type))
            write(save_file, "pn_steps", pn_steps)
            write(save_file, "iterations", iterations_)
        end
        if time_optimal
            # Sample the important metrics.
            (c_sample, d2c_sample, et_sample) = sample_controls(save_file_path)
            h5open(save_file_path, "r+") do save_file
                write(save_file, "controls_sample", c_sample)
                write(save_file, "d2controls_sample", d2c_sample)
                write(save_file, "evolution_time_sample", et_sample)
            end
        end
    end
end

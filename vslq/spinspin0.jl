"""
spinspin0.jl
"""

using Altro
using HDF5
using LinearAlgebra
using RobotDynamics
using StaticArrays
using TrajectoryOptimization

WDIR = get(ENV, "QOCE_PATH", "..")
EXPERIMENT_META = "vslq"
EXPERIMENT_NAME = "spinspin0"

include(joinpath(WDIR, EXPERIMENT_META, "common.jl"))


# paths
SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# Define the optimization.
CONSTRAINT_TOLERANCE = 1e-8
CONTROL_COUNT = 2
DT_STATIC = DT_PREF
DT_STATIC_INV = DT_PREF_INV
DT_INIT = DT_PREF
DT_INIT_INV = DT_PREF_INV
DT_MIN = DT_INIT / 2
DT_MAX = DT_INIT * 2
EVOLUTION_TIME = 20.

# Define the problem.
INITIAL_STATE_1 = II_ISO_1
INITIAL_STATE_2 = II_ISO_2
INITIAL_STATE_3 = II_ISO_3
INITIAL_STATE_4 = II_ISO_4
STATE_SIZE, = size(INITIAL_STATE_1)
INITIAL_ASTATE = [
    INITIAL_STATE_1;
    # INITIAL_STATE_2;
    # INITIAL_STATE_3;
    # INITIAL_STATE_4;
    SizedVector{CONTROL_COUNT}(zeros(CONTROL_COUNT)); # control
    SizedVector{CONTROL_COUNT}(zeros(CONTROL_COUNT)); # dcontrol_dt
]
ASTATE_SIZE, = size(INITIAL_ASTATE)
TARGET_STATE_1 = XPIBY2I_ISO_1
TARGET_STATE_2 = XPIBY2I_ISO_2
TARGET_STATE_3 = XPIBY2I_ISO_3
TARGET_STATE_4 = XPIBY2I_ISO_4
TARGET_ASTATE = [
    TARGET_STATE_1;
    # TARGET_STATE_2;
    # TARGET_STATE_3;
    # TARGET_STATE_4;
    SizedVector{CONTROL_COUNT}(zeros(CONTROL_COUNT)); # control
    SizedVector{CONTROL_COUNT}(zeros(CONTROL_COUNT)); # dcontrol_dt
]
# state indices
STATE_1_IDX = 1:STATE_SIZE
# STATE_2_IDX = STATE_1_IDX[end] + 1:STATE_1_IDX[end] + STATE_SIZE
# STATE_3_IDX = STATE_2_IDX[end] + 1:STATE_2_IDX[end] + STATE_SIZE
# STATE_4_IDX = STATE_3_IDX[end] + 1:STATE_3_IDX[end] + STATE_SIZE
# CONTROLS_IDX = STATE_4_IDX[end] + 1:STATE_4_IDX[end] + CONTROL_COUNT
# DCONTROLS_DT_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
CONTROLS_IDX = STATE_1_IDX[end] + 1:STATE_1_IDX[end] + CONTROL_COUNT
DCONTROLS_DT_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
# control indices
D2CONTROLS_DT2_IDX = 1:CONTROL_COUNT
DT_IDX = D2CONTROLS_DT2_IDX[end] + 1:D2CONTROLS_DT2_IDX[end] + 1


# Specify logging.
VERBOSE = true
SAVE = true

struct Model <: AbstractModel
    n :: Int
    m :: Int
end


Base.size(model::Model) = (model.n, model.m)


function run_traj(;time_optimal=false)
    # Choose dynamics
    if time_optimal
        eval(:(
            function RobotDynamics.dynamics(model::Model, astate, acontrols, time)
            negi_h = (
                NEGI_H0_ISO
                + astate[CONTROLS_IDX][1] * NEGI_H1_ISO
                + astate[CONTROLS_IDX][2] * NEGI_H2_ISO
            )
            delta_state_1 = negi_h * astate[STATE_1_IDX]
            # delta_state_2 = negi_h * astate[STATE_2_IDX]
            # delta_state_3 = negi_h * astate[STATE_3_IDX]
            # delta_state_4 = negi_h * astate[STATE_4_IDX]
            delta_control = astate[DCONTROLS_DT_IDX]
            delta_dcontrol_dt = acontrols[D2CONTROLS_DT2_IDX]
            return [
                delta_state_1;
                # delta_state_2;
                # delta_state_3;
                # delta_state_4;
                delta_control;
                delta_dcontrol_dt;
            ] .* acontrols[DT_IDX][1]^2
            end
        ))
    else
        eval(:(
            function RobotDynamics.dynamics(model::Model, astate, acontrols, time)
            negi_h = (
                NEGI_H0_ISO
                + astate[CONTROLS_IDX][1] * NEGI_H1_ISO
                + astate[CONTROLS_IDX][2] * NEGI_H2_ISO
            )
            delta_state_1 = negi_h * astate[STATE_1_IDX]
            # delta_state_2 = negi_h * astate[STATE_2_IDX]
            # delta_state_3 = negi_h * astate[STATE_3_IDX]
            # delta_state_4 = negi_h * astate[STATE_4_IDX]
            delta_control = astate[DCONTROLS_DT_IDX]
            delta_dcontrol_dt = acontrols[D2CONTROLS_DT2_IDX]
            return [
                delta_state_1;
                # delta_state_2;
                # delta_state_3;
                # delta_state_4;
                delta_control;
                delta_dcontrol_dt;
            ]
            end
        ))
    end
    # Convert to trajectory optimization language.
    n = ASTATE_SIZE
    t0 = 0.
    tf = EVOLUTION_TIME
    x0 = INITIAL_ASTATE
    xf = TARGET_ASTATE
    if time_optimal
        m = CONTROL_COUNT + 1
        dt = 1
        N = Int(floor(EVOLUTION_TIME * DT_INIT_INV)) + 1
    else
        m = CONTROL_COUNT
        dt = DT_STATIC
        N = Int(floor(EVOLUTION_TIME * DT_STATIC_INV)) + 1
    end
    
    # Bound the control amplitude.
    x_max = SizedVector{n}([
        # fill(Inf, 4 * STATE_SIZE);
        fill(Inf, STATE_SIZE);
        fill(MAX_AMP_1, CONTROL_COUNT); # control
        fill(Inf, CONTROL_COUNT);
    ])
    x_min = SizedVector{n}([
        # fill(-Inf, 4 * STATE_SIZE);
        fill(-Inf, STATE_SIZE);
        fill(-MAX_AMP_1, CONTROL_COUNT); # control
        fill(-Inf, CONTROL_COUNT);
    ])
    # Controls start and end at 0.
    x_max_boundary = SizedVector{n}([
        # fill(Inf, 4 * STATE_SIZE);
        fill(Inf, STATE_SIZE);
        fill(0, CONTROL_COUNT); # control
        fill(Inf, CONTROL_COUNT);
    ])
    x_min_boundary = SizedVector{n}([
        # fill(-Inf, 4 * STATE_SIZE);
        fill(-Inf, STATE_SIZE);
        fill(0, CONTROL_COUNT); # control
        fill(-Inf, CONTROL_COUNT);
    ])
    # Bound dt.
    if time_optimal
        u_min = SizedVector{m}([
            fill(-Inf, CONTROL_COUNT);
            fill(sqrt(DT_MIN), 1); # dt
        ])
        u_max = SizedVector{m}([
            fill(Inf, CONTROL_COUNT);
            fill(sqrt(DT_MAX), 1); # dt
        ])
    else
        u_min = SizedVector{m}([
            fill(-Inf, CONTROL_COUNT);
        ])
        u_max = SizedVector{m}([
            fill(Inf, CONTROL_COUNT);
        ])
    end

    # Generate initial trajectory.
    model = Model(n, m)
    if time_optimal
        U0 = [SizedVector{m}([
            fill(1e-4, CONTROL_COUNT);
            fill(DT_INIT, 1);
        ]) for k = 1:N - 1]
    else
        U0 = [SizedVector{m}(
            fill(1e-4, CONTROL_COUNT)
        ) for k = 1:N - 1]
    end
    X0 = [SizedVector{n}([
        fill(1., 1);
        fill(0., n-1);
    ]) for k = 1:N]
    Z = Traj(X0, U0, dt * ones(N))

    # Define penalties.
    Q = Diagonal(SizedVector{n}([
        # fill(1e-1, 4 * STATE_SIZE); # states
        fill(1e-1, STATE_SIZE); # states
        fill(1e-1, CONTROL_COUNT); # control
        fill(1e-1, CONTROL_COUNT); # dcontrol_dt
    ]))
    Qf = Q * N
    if time_optimal
        R = Diagonal(SizedVector{m}([
            fill(1e-1, CONTROL_COUNT); # d2control_dt2
            fill(1e3, 1); # dt
        ]))
    else
        R = Diagonal(SizedVector{m}([
            fill(1e-1, CONTROL_COUNT); # d2control_dt2
        ]))
    end
    obj = LQRObjective(Q, R, Qf, xf, N)

    # Must satisfy control amplitude bound.
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # Must statisfy conrols start and stop at 0.
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # Must satisfy dt bound.
    dt_bnd = BoundConstraint(n, m, u_max=u_max, u_min=u_min)
    # Must reach target state.
    # target_astate_constraint = GoalConstraint(xf, [STATE_1_IDX; STATE_2_IDX; STATE_3_IDX; STATE_4_IDX])
    target_astate_constraint = GoalConstraint(xf, Array(STATE_1_IDX))
    # Must obey unit norm.
    normalization_constraint_1 = NormConstraint(n, m, 1, TrajectoryOptimization.Equality(), STATE_1_IDX)
    # normalization_constraint_2 = NormConstraint(n, m, 1, TrajectoryOptimization.Equality(), STATE_2_IDX)
    
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, 1:1)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N)
    add_constraint!(constraints, normalization_constraint_1, 1:N)
    if time_optimal
        add_constraint!(constraints, dt_bnd, 1:N-1)
    end

    # Instantiate problem and solve.
    prob = Problem{RobotDynamics.RK4}(model, obj, constraints, x0, xf, Z, N, t0, tf)
    opts = SolverOptions(verbose=VERBOSE)
    solver = AugmentedLagrangianSolver(prob, opts)
    solver.opts.constraint_tolerance = CONSTRAINT_TOLERANCE
    solver.opts.constraint_tolerance_intermediate = CONSTRAINT_TOLERANCE
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
    d2cdt2idx_arr = Array(D2CONTROLS_DT2_IDX)
    dtidx_arr = Array(DT_IDX)
    # Square the dts.
    if time_optimal
        acontrols_arr[:, DT_IDX] = acontrols_arr[:, DT_IDX] .^2
    end
    
    # Save.
    if SAVE
        save_file_path = generate_save_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
        println("Saving this optimization to $(save_file_path)")
        h5open(save_file_path, "cw") do save_file
            write(save_file, "acontrols", acontrols_arr)
            write(save_file, "controls_idx", cidx_arr)
            write(save_file, "d2controls_dt2_idx", d2cdt2idx_arr)
            write(save_file, "dt_idx", dtidx_arr)
            write(save_file, "evolution_time", tf)
            write(save_file, "astates", astates_arr)
            write(save_file, "Q", Q_arr)
            write(save_file, "Qf", Qf_arr)
            write(save_file, "R", R_arr)
        end
        if time_optimal
            (controls_sample, d2controls_dt2_sample, evolution_time_sample) = sample_controls(save_file_path)
            h5open(save_file_path, "r+") do save_file
                write(save_file, "controls_sample", controls_sample)
                write(save_file, "d2controls_dt2_sample", d2controls_dt2_sample)
                write(save_file, "evolution_time_sample", evolution_time_sample)
            end
        end
    end
end

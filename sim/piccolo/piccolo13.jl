"""
piccolo13.jl - vanilla trajectory optimization on the piccolo system
"""

# `using Package` is like `from Package import *`in Python.
# `import Package` is like `import Package` in Python.
# `import Package: function` is like `from Package import function` in Python.
# Importing is a one-time cost so importing the
# entire package isn't a big deal unless there are name-space conflicts,
# e.g. two functions have the same name.
using HDF5
using LinearAlgebra
using TrajectoryOptimization
import Plots
using Printf
using StaticArrays

# Construct paths.
EXPERIMENT_META = "piccolo"
EXPERIMENT_NAME = "piccolo13"
WDIR = get(ENV, "MULTIMODE_QOC_PATH", ".")
SAVE_PATH = joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME)

# Plotting configuration.
ENV["GKSwstype"] = "nul"
Plots.gr()
DPI = 300


"""
generate_save_file_path - Generate a unique save file path in the directory
    `save_path` for an h5 file with a numeric prefix and suffix `save_file_name`.

Arguments:
save_file_name :: String - The suffix for the file.
save_path :: String - Directory to save the file to.

Returns:
save_file_path :: String - The unique save file path.

Notes:
This function is not thread safe, i.e. its correctness is not guaranteed
when executed concurrently.
"""
function generate_save_file_path(save_file_name, save_path)
    # Ensure the path exists.
    mkpath(save_path)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for (_, _, files) in walkdir(save_path)
        for file_name in files
            if occursin("_$save_file_name.h5", file_name)
                max_numeric_prefix = max(parse(Int, split(file_name, "_")[1]))
            end
        end
    end

    save_file_name = "_$save_file_name.h5"
    save_file_name = @sprintf("%05d%s", max_numeric_prefix + 1, save_file_name)

    return joinpath(save_path, save_file_name)
end


"""
plot_controls - Plot the controls.
"""
function plot_controls(controls_file_path, save_file_path;
                       controls_idx=CONTROLS_IDX,
                       labels=nothing, title=nothing)
    # Grab and prep data.
    (
        controls,
        evolution_time,
        states,
    ) = h5open(controls_file_path, "r+") do save_file
        controls = read(save_file, "controls")
        evolution_time = read(save_file, "evolution_time")
        states = read(save_file, "states")
        return (
            controls,
            evolution_time,
            states
        )
    end
    (control_eval_count, control_count) = size(controls)
    control_eval_times = Array(range(0., stop=evolution_time, length=control_eval_count))
    file_name = split(basename(controls_file_path), ".h5")[1]
    if isnothing(title)
        title = file_name
    end
    controls_ = states[1:end - 1, controls_idx]

    # Plot.
    fig = Plots.plot(control_eval_times, controls_, show=false, dpi=DPI,
                     title=title, labels=labels)
    Plots.xlabel!("Time (ns)")
    Plots.ylabel!("Amplitude (GHz)")
    Plots.savefig(fig, save_file_path)
    return
end


"""
mat_to_isomat - Take a complex matrix and transform it into
    a matrix in the complex to real isomorphism.

Arguments:
mat :: n x n Array{T, 2} - The complex matrix to convert.

Returns:
isomat :: 2n x 2n Array{T, 2} - `mat` in the isomorphism.
"""
function mat_to_isomat(mat)
    mat_r = real(mat)
    mat_i = imag(mat)
    return [
        mat_r -mat_i;
        mat_i mat_r
    ]
end


"""
vec_to_isovec - Take a complex vector and transform it into
    a vector in the complex to real isomorphism.

Arguments:
vec :: n Array{T, 1} - The complex vector to convert.

Returns:
isovec :: 2n Array{T, 1} - `vec` in the isomorphism.
"""
function vec_to_isovec(vec)
    return [
        real(vec);
        imag(vec)
    ]
end


# Define experimental constants.
CAVITY_FREQ = 2 * pi * 4.4526
KAPPA = 2 * pi * -2.82e-6
TRANSMON_FREQ = 2 * pi * 5.6640
ALPHA = 2 * pi * -1.395126e-1
CHI_E = 2 * pi * -5.64453e-4
CHI_E_2 = 2 * pi * -7.3e-7
MAX_AMP_NORM_CAVITY = sqrt(2) * 2 * pi * 4e-4
MAX_AMP_NORM_TRANSMON = sqrt(2) * 2 * pi * 4e-3

# Define the system.
TRANSMON_STATE_COUNT = 2
CAVITY_STATE_COUNT = 2
HILBERT_DIM = TRANSMON_STATE_COUNT * CAVITY_STATE_COUNT
HILBERT_DIM_ISO = 2 * HILBERT_DIM

# This is like np.linspace(0, TRANSMON_STATE_COUNT - 1, TRANSMON_STATE_COUNT)
# The Array{Int64, 1}(...) is casting it to a 1-dimensional array of 64-bit integers.
TRANSMON_STATES = Array{Int64, 1}(range(0, stop=TRANSMON_STATE_COUNT - 1,
                                        length=TRANSMON_STATE_COUNT))
# This is placing sqrt(1), ..., sqrt(TRANSMON_STATE_COUNT - 2) on the 1st diagonal
# counted up from the true diagonal.
TRANSMON_ANNIHILATE = diagm(1 => map(sqrt, TRANSMON_STATES[1:end - 1]))
# Taking the adjoint is as simple as adding an apostrophe.
TRANSMON_CREATE = TRANSMON_ANNIHILATE'
# In julia `a * b` is scalar multiplication where a and b are scalars,
# `a * b` is matrix multiplication where a and b are vectors,
# and `a .* b` is the element-wise product where a and b are vectors.
TRANSMON_NUMBER = TRANSMON_CREATE * TRANSMON_ANNIHILATE
TRANSMON_ID = I(TRANSMON_STATE_COUNT)
TRANSMON_VACUUM = zeros(TRANSMON_STATE_COUNT)
# Julia is 1-indexed, that means the first element in the
# array is `arr[1]` not like `arr[0]` in python.
# To get the last element of the array you do `arr[end]`
# or `arr[end - 1]` which is like `arr[-1]` or `arr[-2]` in Python.
TRANSMON_G = copy(TRANSMON_VACUUM)
TRANSMON_G[1] = 1
TRANSMON_E = copy(TRANSMON_VACUUM)
TRANSMON_E[2] = 1

CAVITY_STATES = Array{Int64, 1}(range(0, stop=CAVITY_STATE_COUNT - 1, length=CAVITY_STATE_COUNT))
CAVITY_ANNIHILATE = diagm(1 => map(sqrt, CAVITY_STATES[1:end - 1]))
CAVITY_CREATE = CAVITY_ANNIHILATE'
CAVITY_NUMBER = CAVITY_CREATE * CAVITY_ANNIHILATE
CAVITY_ID = I(CAVITY_STATE_COUNT)
CAVITY_QUAD = CAVITY_NUMBER * (CAVITY_NUMBER - CAVITY_ID)
CAVITY_VACUUM = zeros(CAVITY_STATE_COUNT)
CAVITY_ZERO = copy(CAVITY_VACUUM)
CAVITY_ZERO[1] = 1
CAVITY_ONE = copy(CAVITY_VACUUM)
CAVITY_ONE[2] = 1

# Static hamiltonian.
# You will see calls to `SMatrix`, `SVector`, `@SVector`, etc.
# These cast the arrays to StaticArrays, as in the name of the package imported above.
# Static arrays have performance advantages compared to fixed arrays, namely
# their size is known by the compiler ahead of time, hence the specificiation
# that `H_S` is a matrix of size HILBERT_DIM_ISO x HILBERT_DIM_ISO.
# We define all of our hamiltonians as -1im * H so that when we write
# the schroedinger equation dynamics we don't have to do any extra
# multiplication.
NEG_I_H_S = SMatrix{HILBERT_DIM_ISO, HILBERT_DIM_ISO}(mat_to_isomat(
    - 1im * (
        # CAVITY_FREQ * kron(TRANSMON_ID, CAVITY_NUMBER)
        + (KAPPA / 2) * kron(TRANSMON_ID, CAVITY_QUAD)
        # + TRANSMON_FREQ * kron(TRANSMON_NUMBER, CAVITY_ID)
        # + (ALPHA / 2) * kron(TRANSMON_QUAD, CAVITY_ID)
        + 2 * CHI_E * kron(TRANSMON_E * TRANSMON_E', CAVITY_NUMBER)
        + CHI_E_2 * kron(TRANSMON_E * TRANSMON_E', CAVITY_QUAD)
    )
))

# Control hamiltonians.
# Complex automatic differentiation in julia is not fully supported yet,
# so we do the same u(t) a + conj(u(t)) a_dagger where u = ux + i uy
# business.
# Transmon drive.
NEG_I_H_C1_R = SMatrix{HILBERT_DIM_ISO, HILBERT_DIM_ISO}(mat_to_isomat(
    -1im * kron(TRANSMON_G * TRANSMON_E' + TRANSMON_E * TRANSMON_G', CAVITY_ID)
))
NEG_I_H_C1_I = SMatrix{HILBERT_DIM_ISO, HILBERT_DIM_ISO}(mat_to_isomat(
    -1im * kron(1im * (TRANSMON_G * TRANSMON_E' - TRANSMON_E * TRANSMON_G'), CAVITY_ID)
))
# Cavity drive.
NEG_I_H_C2_R = SMatrix{HILBERT_DIM_ISO, HILBERT_DIM_ISO}(mat_to_isomat(
    -1im * kron(TRANSMON_ID, CAVITY_ANNIHILATE + CAVITY_CREATE)
))
NEG_I_H_C2_I = SMatrix{HILBERT_DIM_ISO, HILBERT_DIM_ISO}(mat_to_isomat(
    -1im * kron(TRANSMON_ID, 1im * (CAVITY_ANNIHILATE - CAVITY_CREATE))
))

# Define the optimization.
EVOLUTION_TIME = 5e2 #ns
CONTROL_COUNT = 4
DT = 1e-2
N = Int(EVOLUTION_TIME / DT) + 1
ITERATION_COUNT = Int(1e3)

# Define the problem.
INITIAL_STATE = SVector{HILBERT_DIM_ISO}(vec_to_isovec(
    kron(TRANSMON_G, CAVITY_ZERO)
))
STATE_SIZE, = size(INITIAL_STATE)
INITIAL_ASTATE = [
    INITIAL_STATE; # state
    @SVector zeros(CONTROL_COUNT); # int_control
    @SVector zeros(CONTROL_COUNT); # control
    @SVector zeros(CONTROL_COUNT); # dcontrol_dt
]
ASTATE_SIZE, = size(INITIAL_ASTATE)
TARGET_STATE = SVector{HILBERT_DIM_ISO}(vec_to_isovec(
    kron(TRANSMON_G, CAVITY_ONE)
))
TARGET_ASTATE = [
    TARGET_STATE;
    @SVector zeros(CONTROL_COUNT);
    @SVector zeros(CONTROL_COUNT);
    @SVector zeros(CONTROL_COUNT);
]
STATE_IDX = 1:STATE_SIZE
INT_CONTROLS_IDX = STATE_IDX[end] + 1:STATE_IDX[end] + CONTROL_COUNT
CONTROLS_IDX = INT_CONTROLS_IDX[end] + 1:INT_CONTROLS_IDX[end] + CONTROL_COUNT
DCONTROLS_DT_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT

# Generate initial controls.
GRAB_CONTROLS = false
INITIAL_CONTROLS = nothing
if GRAB_CONTROLS
    controls_file_path = joinpath(SAVE_PATH, "00000_piccolo13.h5")
    INITIAL_CONTROLS = h5open(controls_file_path, "r") do save_file
        controls = Array(save_file["controls"])
        return [
            SVector{CONTROL_COUNT}(controls[i]) for i = 1:N-1
        ]
    end
else
    # INIITAL_CONTROLS should be small if optimizing over derivatives.
    INITIAL_CONTROLS = [
        @SVector fill(1e-4, CONTROL_COUNT) for k = 1:N-1
    ]
end

# Specify logging.
VERBOSE = true
SAVE = true

struct Model <: AbstractModel
    n :: Int
    m :: Int
end


function Base.size(model::Model)
    return model.n, model.m
end


function TrajectoryOptimization.dynamics(model::Model, astate, d2controls_dt2, time)
    neg_i_hamiltonian = (
        NEG_I_H_S
        + astate[CONTROLS_IDX][1] * NEG_I_H_C1_R
        + astate[CONTROLS_IDX][2] * NEG_I_H_C1_I
        + astate[CONTROLS_IDX][3] * NEG_I_H_C2_R
        + astate[CONTROLS_IDX][4] * NEG_I_H_C2_I
    )
    # delta_state is the lhs of the schroedinger equation
    delta_state = neg_i_hamiltonian * astate[STATE_IDX]
    delta_int_control = astate[CONTROLS_IDX]
    delta_control = astate[DCONTROLS_DT_IDX]
    delta_dcontrol_dt = d2controls_dt2
    return [
        delta_state;
        delta_int_control;
        delta_control;
        delta_dcontrol_dt;
    ]
end


function run_traj()
    dt = DT
    n = ASTATE_SIZE
    m = CONTROL_COUNT
    t0 = 0.
    tf = EVOLUTION_TIME
    x0 = INITIAL_ASTATE
    xf = TARGET_ASTATE
    # control amplitude constraint
    x_max = [
        @SVector fill(Inf, STATE_SIZE);
        @SVector fill(Inf, CONTROL_COUNT);
        @SVector fill(MAX_AMP_NORM_TRANSMON, 2); # control
        @SVector fill(MAX_AMP_NORM_CAVITY, 2); # control
        @SVector fill(Inf, CONTROL_COUNT);
    ]
    x_min = [
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, CONTROL_COUNT);
        @SVector fill(-MAX_AMP_NORM_TRANSMON, 2); # control
        @SVector fill(-MAX_AMP_NORM_CAVITY, 2); # control
        @SVector fill(-Inf, CONTROL_COUNT);
    ]
    # controls start and end at 0
    x_max_boundary = [
        @SVector fill(Inf, STATE_SIZE);
        @SVector fill(Inf, CONTROL_COUNT);
        @SVector fill(0, CONTROL_COUNT); # control
        @SVector fill(Inf, CONTROL_COUNT);
    ]
    x_min_boundary = [
        @SVector fill(-Inf, STATE_SIZE);
        @SVector fill(-Inf, CONTROL_COUNT);
        @SVector fill(0, CONTROL_COUNT); # control
        @SVector fill(-Inf, CONTROL_COUNT);
    ]

    model = Model(n, m)
    U0 = INITIAL_CONTROLS
    X0 = [
        @SVector fill(NaN, n) for k = 1:N
    ]
    Z = Traj(X0, U0, dt * ones(N))

    Q = Diagonal([
        @SVector fill(1e-1, STATE_SIZE);
        @SVector fill(1e-1, CONTROL_COUNT); # int_control
        @SVector fill(1e-1, CONTROL_COUNT); # control
        @SVector fill(1e-1, CONTROL_COUNT); # dcontrol_dt
    ])
    Qf = Q * N
    R = Diagonal(@SVector fill(1e-1, m))
    obj = LQRObjective(Q, R, Qf, xf, N)

    # must satisfy control amplitudes
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and stop at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have integral of controls = 0
    target_astate_constraint = GoalConstraint(xf, [STATE_IDX;INT_CONTROLS_IDX])
    
    constraints = ConstraintSet(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, 1:1)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N)
    
    prob = Problem{RK3}(model, obj, constraints, x0, xf, Z, N, t0, tf)
    opts = SolverOptions(verbose=VERBOSE)
    solver = ALTROSolver(prob, opts)
    # ilqr_opts = iLQRSolverOptions(verbose=VERBOSE)
    # ilqr_opts.static_bp = false
    # solver = iLQRSolver(prob, ilqr_opts)
    solve!(solver)

    controls_raw = controls(solver)
    controls_arr = permutedims(reduce(hcat, map(Array, controls_raw)))
    states_raw = states(solver)
    states_arr = permutedims(reduce(hcat, map(Array, states_raw)))
    Q_raw = Array(Q)
    Q_arr = [Q_raw[i, i] for i in 1:size(Q_raw)[1]]
    Qf_raw = Array(Qf)
    Qf_arr = [Qf_raw[i, i] for i in 1:size(Qf_raw)[1]]
    R_raw = Array(R)
    R_arr = [R_raw[i, i] for i in 1:size(R_raw)[1]]

    # Save.
    if SAVE
        save_file_path = generate_save_file_path(EXPERIMENT_NAME, SAVE_PATH)
        @printf("Saving this optimization to %s\n", save_file_path)
        h5open(save_file_path, "cw") do save_file
            write(save_file, "controls", controls_arr)
            write(save_file, "evolution_time", tf)
            write(save_file, "states", states_arr)
            write(save_file, "Q", Q_arr)
            write(save_file, "Qf", Qf_arr)
            write(save_file, "R", R_arr)
        end
    end
end

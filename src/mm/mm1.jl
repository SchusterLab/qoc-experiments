"""DSTATE
mm1.jl - first multimode experiment
"""

WDIR = joinpath(@__DIR__, "../..")
include(joinpath(WDIR, "src", "mm", "mm.jl"))

using Altro
using CUDA
using ForwardDiff
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

# model
struct Model{TM,TMd,TV,TVi,Tis,Tic} <: AbstractModel
    # problem size
    n::Int
    m::Int
    # problem
    Hs::Vector{TM}
    derivative_count::Int
    # indices
    state1_idx::Tis
    controls_idx::Tic
    dcontrols_idx::Tic
    dstate1_idx::Tis
    d2controls_idx::Tic
    # tmp
    mtmp::Vector{TM}
    mtmp_dense::Vector{TMd}
    vtmp::Vector{TV}
    ipiv_tmp::TVi
end

function Model(M_, Md_, V_, Hs, derivative_count)
    # problem size
    control_count = 4
    state_count = 1 + derivative_count
    n = state_count * HDIM_ISO + 2 * control_count
    m = control_count
    # state indices
    state1_idx = V(1:HDIM_ISO)
    controls_idx = V(state1_idx[end] + 1:state1_idx[end] + control_count)
    dcontrols_idx = V(controls_idx[end] + 1:controls_idx[end] + control_count)
    dstate1_idx = V(dcontrols_idx[end] + 1:dcontrols_idx[end] + HDIM_ISO)
    # control indices
    d2controls_idx = V(1:control_count)
    # temporary stores
    mtmp = [M_(zeros(HDIM_ISO, HDIM_ISO)) for i = 1:34]
    mtmp_dense = [Md_(zeros(HDIM_ISO, HDIM_ISO)) for i = 1:2]
    vtmp = [V_(zeros(HDIM_ISO)) for i = 1:4]
    ipiv_tmp = V_(zeros(Int, HDIM_ISO))
    if ipiv_tmp isa CuVector
        ipiv_tmp = V_(zeros(Int32, HDIM_ISO))
    end
    # types
    TM = typeof(mtmp[1])
    TMd = typeof(mtmp_dense[1])
    TV = typeof(vtmp[1])
    TVi = typeof(ipiv_tmp)
    Tis = typeof(state1_idx)
    Tic = typeof(controls_idx)
    return Model{TM,TMd,TV,TVi,Tis,Tic}(n, m, Hs, derivative_count, state1_idx, controls_idx,
                                        dcontrols_idx, dstate1_idx, d2controls_idx,
                                        mtmp, mtmp_dense, vtmp, ipiv_tmp)
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
                              acontrol::AbstractVector, time::Real, dt::Real)
    # get hamiltonian and unitary
    H = dt * (
        model.Hs[1]
        + astate[model.controls_idx[1]] * model.Hs[2]
        + astate[model.controls_idx[2]] * model.Hs[3]
        + astate[model.controls_idx[3]] * model.Hs[4]
        + astate[model.controls_idx[4]] * model.Hs[5]
    )
    U = exp(H)
    # propagate state
    state1 = U * astate[model.state1_idx]
    # propagate dstate
    if model.derivative_count == 1
        dstate1 = U * (astate[model.dstate1_idx] + dt * model.Hs[6] * astate[model.state1_idx])
    end
    # propagate controls
    controls = astate[model.dcontrols_idx] .* dt + astate[model.controls_idx]
    # propagate dcontrols
    dcontrols = acontrol[model.d2controls_idx] .* dt + astate[model.dcontrols_idx]
    if model.derivative_count == 1
        astate_ = [state1; controls; dcontrols; dstate1]
    else
        astate_ = [state1; controls; dcontrols]
    end
    return astate_
end

function RD.discrete_dynamics!(astate_::AbstractVector, ::Type{EXP}, model::Model,
                               astate::AbstractVector,
                               acontrol::AbstractVector, time::Real, dt::Real)
    # get hamiltonian and unitary
    H = model.mtmp[29]
    H1r = model.mtmp[30] .= model.Hs[2]
    lmul!(astate[model.controls_idx[1]], H1r)
    H1i = model.mtmp[31] .= model.Hs[3]
    lmul!(astate[model.controls_idx[2]], H1i)
    H2r = model.mtmp[32] .= model.Hs[4]
    lmul!(astate[model.controls_idx[3]], H2r)
    H2i = model.mtmp[33] .= model.Hs[5]
    lmul!(astate[model.controls_idx[4]], H2i)
    for i in eachindex(H)
        H[i] = model.Hs[1][i] + H1r[i] + H1i[i] + H2r[i] + H2i[i]
    end
    lmul!(dt, H)
    U = exp!(model.mtmp, model.mtmp_dense, model.ipiv_tmp, H)
    # propagate state
    state1 = astate[model.state1_idx]
    mul!(model.vtmp[1], U, state1)
    astate_[model.state1_idx] .= model.vtmp[1]
    # propagate dstate
    if model.derivative_count == 1
        mul!(model.vtmp[1], model.Hs[6], state1)
        for i = 1:HDIM_ISO
            model.vtmp[1][i] = dt * model.vtmp[1][i] + astate[model.dstate1_idx[i]]
        end
        mul!(model.vtmp[2], U, model.vtmp[1])
        astate_[model.dstate1_idx] .= model.vtmp[2]
    end
    # propagate controls
    astate_[model.controls_idx] .= astate[model.dcontrols_idx]
    astate_[model.controls_idx] .*= dt
    astate_[model.controls_idx] .+= astate[model.controls_idx]
    # propagate dcontrols
    astate_[model.dcontrols_idx] .= acontrol[model.d2controls_idx]
    astate_[model.dcontrols_idx] .*= dt
    astate_[model.dcontrols_idx] .+= astate[model.dcontrols_idx]
    return nothing
end

function RD.discrete_jacobian!(D::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix,
                               ::Type{EXP}, model::Model, astate::AbstractVector,
                               acontrol::AbstractVector, time::Real, dt::Real,
                               ix::AbstractVector, iu::AbstractVector)
    # get hamiltonian and unitary
    H = model.mtmp[29]
    H1r = model.mtmp[30] .= model.Hs[2]
    lmul!(astate[model.controls_idx[1]], H1r)
    H1i = model.mtmp[31] .= model.Hs[3]
    lmul!(astate[model.controls_idx[2]], H1i)
    H2r = model.mtmp[32] .= model.Hs[4]
    lmul!(astate[model.controls_idx[3]], H2r)
    H2i = model.mtmp[33] .= model.Hs[5]
    lmul!(astate[model.controls_idx[4]], H2i)
    for i in eachindex(H)
        H[i] = model.Hs[1][i] + H1r[i] + H1i[i] + H2r[i] + H2i[i]
    end
    lmul!(dt, H)
    U = exp!(model.mtmp, model.mtmp_dense, model.ipiv_tmp, H)
    # get state at this time step and next
    state1k = model.vtmp[1] .= astate[model.state1_idx]
    state1kp = model.vtmp[2]
    mul!(state1kp, U, state1k)
    # state1, dstate1 modifications
    # s1
    A[model.state1_idx, model.state1_idx] = U
    if model.derivative_count == 1
        mul!(model.mtmp[34], U, model.Hs[6])
        lmul!(dt, model.mtmp[34])
        A[model.dstate1_idx, model.state1_idx] .= model.mtmp[34]
    end
    # c1
    H1r .= model.Hs[2]
    lmul!(dt, H1r)
    dU_dc1 = exp_frechet!(model.mtmp, model.mtmp_dense, model.ipiv_tmp, H, H1r; reuse=true)
    mul!(model.vtmp[3], dU_dc1, state1k)
    A[model.state1_idx, model.controls_idx[1]] .= model.vtmp[3]
    if model.derivative_count == 1
        mul!(model.vtmp[4], model.Hs[6], state1k)
        lmul!(dt, model.vtmp[4])
        mul!(model.vtmp[3], dU_dc1, model.vtmp[4])
        A[model.dstate1_idx, model.controls_idx[1]] .= model.vtmp[3]
    end
    # c2
    H1i .= model.Hs[3]
    lmul!(dt, H1i)
    dU_dc2 = exp_frechet!(model.mtmp, model.mtmp_dense, model.ipiv_tmp, H, H1i; reuse=true)
    mul!(model.vtmp[3], dU_dc2, state1k)
    A[model.state1_idx, model.controls_idx[2]] .= model.vtmp[3]
    if model.derivative_count == 1
        mul!(model.vtmp[3], dU_dc2, model.vtmp[4])
        A[model.dstate1_idx, model.controls_idx[2]] .= model.vtmp[3]
    end
    # c3
    H2r .= model.Hs[4]
    lmul!(dt, H2r)
    dU_dc3 = exp_frechet!(model.mtmp, model.mtmp_dense, model.ipiv_tmp, H, H2r; reuse=true)
    mul!(model.vtmp[3], dU_dc3, state1k)
    A[model.state1_idx, model.controls_idx[3]] .= model.vtmp[3]
    if model.derivative_count == 1
        mul!(model.vtmp[3], dU_dc3, model.vtmp[4])
        A[model.dstate1_idx, model.controls_idx[3]] .= model.vtmp[3]
    end
    # c4
    H2i .= model.Hs[5]
    lmul!(dt, H2i)
    dU_dc4 = exp_frechet!(model.mtmp, model.mtmp_dense, model.ipiv_tmp, H, H2i; reuse=true)
    mul!(model.vtmp[3], dU_dc4, state1k)
    A[model.state1_idx, model.controls_idx[4]] .= model.vtmp[3]
    if model.derivative_count == 1
        mul!(model.vtmp[3], dU_dc4, model.vtmp[4])
        A[model.dstate1_idx, model.controls_idx[4]] .= model.vtmp[3]
    end
    # ds1
    if model.derivative_count == 1
        A[model.dstate1_idx, model.dstate1_idx] = U
    end
    for i = 1:model.m
        # control modifications
        A[model.controls_idx[i], model.controls_idx[i]] = 1
        A[model.controls_idx[i], model.dcontrols_idx[i]] = dt
        # dcontrol modifications
        A[model.dcontrols_idx[i], model.dcontrols_idx[i]] = 1
        B[model.dcontrols_idx[i], model.d2controls_idx[i]] = dt
    end
end


function run_traj(;fock_state=0, evolution_time=2000., dt=1., verbose=true,
                  derivative_count=0, qs=[1e0, 1e-1, 1e-1, 1e-1, 1e-1], smoke_test=false,
                  pn_steps=2, max_penalty=1e11, save=true, max_iterations=Int64(2e5),
                  max_cost=1e8, benchmark=false, ilqr_ctol=1e-2, ilqr_gtol=1e-4,
                  ilqr_max_iterations=300, penalty_scaling=10.)
    Hs = [M(H) for H in (NEGI_H0ROT_ISO, NEGI_H1R_ISO, NEGI_H1I_ISO, NEGI_H2R_ISO, NEGI_H2I_ISO,
                         NEGI_DH0_ISO)]
    model = Model(M, Md, V, Hs, derivative_count)
    n_, m_ = size(model)
    t0 = 0.

    # initial state
    x0 = zeros(n_)
    x0[model.state1_idx] = IS1_ISO
    x0 = V(x0)

    # target state
    xf = zeros(n_)
    cavity_state = zeros(CAVITY_STATE_COUNT)
    cavity_state[fock_state + 1] = 1
    xf[model.state1_idx] = get_vec_iso(kron(cavity_state, TRANSMON_G))
    xf = V(xf)

    # control amplitude constraint at boundary
    x_max = fill(Inf, n_)
    x_max[model.controls_idx[1:2]] .= MAX_AMP_NORM_TRANSMON
    x_max[model.controls_idx[3:4]] .= MAX_AMP_NORM_CAVITY
    x_max = V(x_max)
    u_max = V(fill(Inf, m_))
    x_min = fill(-Inf, n_)
    x_min[model.controls_idx[1:2]] .= -MAX_AMP_NORM_TRANSMON
    x_min[model.controls_idx[3:4]] .= -MAX_AMP_NORM_CAVITY
    x_min = V(x_min)
    u_min = V(fill(-Inf, m_))
    # control amplitude constraint at boundary
    x_max_boundary = fill(Inf, n_)
    x_max_boundary[model.controls_idx] .= 0
    x_max_boundary = V(x_max_boundary)
    u_max_boundary = V(fill(Inf, m_))
    x_min_boundary = fill(-Inf, n_)
    x_min_boundary[model.controls_idx] .= 0
    x_min_boundary = V(x_min_boundary)
    u_min_boundary = V(fill(-Inf, m_))

    # initial trajectory
    N_ = Int(floor(evolution_time / dt)) + 1
    X0 = [V(zeros(n_)) for k = 1:N_]
    X0[1] .= x0
    U0 = [V([
        fill(1e-6, 2);
        fill(1e-6, 2);
    ]) for k = 1:N_-1]
    ts = V(zeros(N_))
    ts[1] = t0
    for k = 1:N_-1
        ts[k + 1] = ts[k] + dt
        RD.discrete_dynamics!(X0[k + 1], EXP, model, X0[k], U0[k], ts[k], dt)
    end
    
    # cost function
    Q = V(zeros(n_))
    Q[model.state1_idx] .= qs[1]
    Q[model.controls_idx] .= qs[2]
    Q[model.dcontrols_idx] .= qs[3]
    if model.derivative_count == 1
        Q[model.dstate1_idx] .= qs[4]
    end
    Q = Diagonal(Q)
    Qf = Q * N_
    R = V(zeros(m_))
    R[model.d2controls_idx] .= qs[4]
    R = Diagonal(R)
    objective = LQRObjective(Q, Qf, R, xf, n_, m_, N_, M, V)

    # must satisfy control amplitude constraints
    control_amp = TO.BoundConstraint(n_, m_, x_max, x_min, u_max, u_min, M, V)
    # must statisfy controls start and stop at 0
    control_amp_boundary = TO.BoundConstraint(n_, m_, x_max_boundary, x_min_boundary,
                                              u_max_boundary, u_min_boundary, M, V)
    # must reach target state, must have integral of controls = 0
    target_astate_constraint = TO.GoalConstraint(n_, m_, xf, model.state1_idx, M, V)
    # build constraints
    constraints = TO.ConstraintList(n_, m_, N_, M, V)
    TO.add_constraint!(constraints, control_amp, V(2:N_-2))
    TO.add_constraint!(constraints, control_amp_boundary, V(N_-1:N_-1))
    TO.add_constraint!(constraints, target_astate_constraint, V(N_:N_))
    # build problem
    prob = Problem(EXP, model, objective, constraints, X0, U0, ts, N_, M, Md, V)
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
        penalty_scaling=penalty_scaling
    )
    # solve
    solver = ALTROSolver(prob, opts)
    if benchmark
        benchmark_result = Altro.benchmark_solve!(solver)
    else
        benchmark_result = nothing
        Altro.solve!(solver)
    end
    # post-process
    acontrols_raw = Altro.controls(solver)
    acontrols_arr = permutedims(reduce(hcat, map(Array, acontrols_raw)), [2, 1])
    astates_raw = Altro.states(solver)
    astates_arr = permutedims(reduce(hcat, map(Array, astates_raw)), [2, 1])
    Q_raw = Array(Q)
    Q_arr = [Q_raw[i, i] for i in 1:size(Q_raw)[1]]
    Qf_raw = Array(Qf)
    Qf_arr = [Qf_raw[i, i] for i in 1:size(Qf_raw)[1]]
    R_raw = Array(R)
    R_arr = [R_raw[i, i] for i in 1:size(R_raw)[1]]
    cidx_arr = Array(model.controls_idx)
    d2cidx_arr = Array(model.d2controls_idx)
    max_v, max_v_info = Altro.max_violation_info(solver)
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
        "max_v" => max_v,
        "max_v_info" => max_v_info,
        "dt" => dt,
        "ts" => ts,
        "derivative_count" => derivative_count,
        "max_penalty" => max_penalty,
        "ilqr_ctol" => ilqr_ctol,
        "ilqr_gtol" => ilqr_gtol,
        "save_type" => Integer(jl),
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

"""
mm1.jl - first multimode experiment
"""

WDIR = joinpath(@__DIR__, "../..")
include(joinpath(WDIR, "src", "mm", "mm.jl"))

using Altro
using CUDA
using ForwardDiff
using HDF5
using IterTools
using LinearAlgebra
using SparseArrays
using StaticArrays
const FD = ForwardDiff

# paths
const EXPERIMENT_META = "mm"
const EXPERIMENT_NAME = "mm1"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

# model
struct Model{TM,TMd,TV,TVi,Tis,Tic,Tid} <: AbstractModel
    # problem size
    n::Int
    m::Int
    control_count::Int
    # problem
    Hs::Vector{TM}
    derivative_count::Int
    time_optimal::Bool
    # indices
    state1_idx::Tis
    controls_idx::Tic
    dcontrols_idx::Tic
    dstate1_idx::Tis
    d2controls_idx::Tic
    dt_idx::Tid
    # tmp
    mtmp::Vector{TM}
    mtmp_dense::Vector{TMd}
    vtmp::Vector{TV}
    ipiv_tmp::TVi
end

function Model(M_, Md_, V_, Hs, derivative_count, time_optimal)
    # problem size
    control_count = 4
    state_count = 1 + derivative_count
    n = state_count * HDIM_ISO + 2 * control_count
    m = time_optimal ? control_count + 1 : control_count
    # state indices
    state1_idx = V(1:HDIM_ISO)
    controls_idx = V(state1_idx[end] + 1:state1_idx[end] + control_count)
    dcontrols_idx = V(controls_idx[end] + 1:controls_idx[end] + control_count)
    dstate1_idx = V(dcontrols_idx[end] + 1:dcontrols_idx[end] + HDIM_ISO)
    # control indices
    d2controls_idx = V(1:control_count)
    dt_idx = V(d2controls_idx[end] + 1:d2controls_idx[end] + 1)
    # temporary stores
    mtmp = [M_(zeros(HDIM_ISO, HDIM_ISO)) for i = 1:34]
    mtmp_dense = [Md_(zeros(HDIM_ISO, HDIM_ISO)) for i = 1:2]
    vtmp = [V_(zeros(HDIM_ISO)) for i = 1:5]
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
    Tid = typeof(dt_idx)
    return Model{TM,TMd,TV,TVi,Tis,Tic,Tid}(
        n, m, control_count, Hs, derivative_count, time_optimal,
        state1_idx, controls_idx, dcontrols_idx, dstate1_idx, d2controls_idx,
        dt_idx, mtmp, mtmp_dense, vtmp, ipiv_tmp
    )
end

@inline Base.size(model::Model) = model.n, model.m
# vector and matrix constructors (use CPU arrays)
@inline M(mat_) = mat_
@inline Md(mat_) = mat_
@inline V(vec_) = vec_

# dynamics
abstract type EXP <: Altro.Explicit end

function Altro.discrete_dynamics(::Type{EXP}, model::Model,
                              astate::AbstractVector,
                              acontrol::AbstractVector, time::Real, dt_::Real)
    dt = !model.time_optimal ? dt_ : acontrol[model.dt_idx[1]]^2
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
        dstate1 = U * ((astate[model.dstate1_idx] ./ dt) + model.Hs[6] * astate[model.state1_idx])
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

function Altro.discrete_dynamics!(astate_::AbstractVector, ::Type{EXP}, model::Model,
                                  astate::AbstractVector,
                                  acontrol::AbstractVector, time::Real, dt_::Real)
    dt = !model.time_optimal ? dt_ : acontrol[model.dt_idx[1]]^2
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
        model.vtmp[1] .= astate[model.dstate1_idx]
        mul!(model.vtmp[1], model.Hs[6], state1, 1., dt^(-1))
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

function Altro.discrete_jacobian!(A::AbstractMatrix, B::AbstractMatrix,
                                  ::Type{EXP}, model::Model, astate::AbstractVector,
                                  acontrol::AbstractVector, time::Real, dt_::Real)
    dt = !model.time_optimal ? dt_ : acontrol[model.dt_idx[1]]^2
    sqrt_dt = sqrt(dt)
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
        dstate1 = model.vtmp[3] .= astate[model.dstate1_idx]
        mul!(model.mtmp[34], U, model.Hs[6])
        A[model.dstate1_idx, model.state1_idx] .= model.mtmp[34]
    end
    # c1
    H1r .= model.Hs[2]
    lmul!(dt, H1r)
    dU_dc1 = exp_frechet!(model.mtmp, model.mtmp_dense, model.ipiv_tmp, H, H1r; reuse=true)
    mul!(model.vtmp[5], dU_dc1, state1k)
    A[model.state1_idx, model.controls_idx[1]] .= model.vtmp[5]
    if model.derivative_count == 1
        model.vtmp[4] .= dstate1
        mul!(model.vtmp[4], model.Hs[6], state1k, 1., dt^(-1))
        mul!(model.vtmp[5], dU_dc1, model.vtmp[4])
        A[model.dstate1_idx, model.controls_idx[1]] .= model.vtmp[5]
    end
    # c2
    H1i .= model.Hs[3]
    lmul!(dt, H1i)
    dU_dc2 = exp_frechet!(model.mtmp, model.mtmp_dense, model.ipiv_tmp, H, H1i; reuse=true)
    mul!(model.vtmp[5], dU_dc2, state1k)
    A[model.state1_idx, model.controls_idx[2]] .= model.vtmp[5]
    if model.derivative_count == 1
        mul!(model.vtmp[5], dU_dc2, model.vtmp[4])
        A[model.dstate1_idx, model.controls_idx[2]] .= model.vtmp[5]
    end
    # c3
    H2r .= model.Hs[4]
    lmul!(dt, H2r)
    dU_dc3 = exp_frechet!(model.mtmp, model.mtmp_dense, model.ipiv_tmp, H, H2r; reuse=true)
    mul!(model.vtmp[5], dU_dc3, state1k)
    A[model.state1_idx, model.controls_idx[3]] .= model.vtmp[5]
    if model.derivative_count == 1
        mul!(model.vtmp[5], dU_dc3, model.vtmp[4])
        A[model.dstate1_idx, model.controls_idx[3]] .= model.vtmp[5]
    end
    # c4
    H2i .= model.Hs[5]
    lmul!(dt, H2i)
    dU_dc4 = exp_frechet!(model.mtmp, model.mtmp_dense, model.ipiv_tmp, H, H2i; reuse=true)
    mul!(model.vtmp[5], dU_dc4, state1k)
    A[model.state1_idx, model.controls_idx[4]] .= model.vtmp[5]
    if model.derivative_count == 1
        mul!(model.vtmp[5], dU_dc4, model.vtmp[4])
        A[model.dstate1_idx, model.controls_idx[4]] .= model.vtmp[5]
    end
    # ds1
    if model.derivative_count == 1
        A[model.dstate1_idx, model.dstate1_idx] .= U
        A[model.dstate1_idx, model.dstate1_idx] .*= dt^(-1)
    end
    # dt
    if model.time_optimal
        H_dsdt = model.mtmp[34] .= H
        lmul!(2 * sqrt_dt * dt^(-1), H_dsdt)
        dU_dsdt = exp_frechet!(model.mtmp, model.mtmp_dense, model.ipiv_tmp, H, H_dsdt;
                               reuse=true)
        mul!(model.vtmp[5], dU_dsdt, state1k)
        B[model.state1_idx, model.dt_idx] .= model.vtmp[5]
        if model.derivative_count == 1
            mul!(model.vtmp[5], dU_dsdt, model.vtmp[4])
            mul!(model.vtmp[5], U, dstate1, -2 * sqrt_dt^(-3), 1.)
            B[model.dstate1_idx, model.dt_idx] .= model.vtmp[5]
        end
    end
    for i = 1:model.control_count
        # control modifications
        A[model.controls_idx[i], model.controls_idx[i]] = 1
        A[model.controls_idx[i], model.dcontrols_idx[i]] = dt
        if model.time_optimal
            B[model.controls_idx[i], model.dt_idx[1]] = (2 * sqrt_dt *
                                                         astate[model.dcontrols_idx[i]])
        end
        # dcontrol modifications
        A[model.dcontrols_idx[i], model.dcontrols_idx[i]] = 1
        B[model.dcontrols_idx[i], model.d2controls_idx[i]] = dt
        if model.time_optimal
            B[model.dcontrols_idx[i], model.dt_idx[1]] = (2 * sqrt_dt *
                                                          acontrol[model.d2controls_idx[i]])
        end
    end
end


function run_traj(;target_level=0, evolution_time=2000., dt=1., verbose=true,
                  derivative_count=0, time_optimal=false,
                  qs=[1e0, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1], smoke_test=false,
                  pn_steps=2, save=true, max_iterations=Int64(2e5),
                  max_cost=1e12, benchmark=false, ilqr_ctol=1e-2, ilqr_gtol=1e-4,
                  ilqr_max_iterations=300, max_state_value=1e10,
                  max_control_value=1e10, al_max_iterations=30,
                  cnp_levels=CAVITY_STATE_COUNT-1:CAVITY_STATE_COUNT-1,
                  cnp_tol=1e-1, max_penalty=1e11, al_vtol=1e-4)
    # construct model
    Hs = [M(H) for H in (NEGI_H0ROT_ISO, NEGI_H1R_ISO, NEGI_H1I_ISO, NEGI_H2R_ISO, NEGI_H2I_ISO,
                         NEGI_DH0_ISO)]
    model = Model(M, Md, V, Hs, derivative_count, time_optimal)
    n_, m_ = size(model)
    t0 = 0.

    # initial state
    x0 = zeros(n_)
    x0[model.state1_idx] = IS1_ISO
    x0 = V(x0)

    # target state
    xf = zeros(n_)
    cavity_state_ = cavity_state(target_level)
    xf[model.state1_idx] = get_vec_iso(kron(cavity_state_, TRANSMON_G))
    xf = V(xf)

    # initial trajectory
    N_ = Int(floor(evolution_time / dt)) + 1
    X0 = [V(zeros(n_)) for k = 1:N_]
    X0[1] .= x0
    U0 = [V([
        fill(1e-6, 2);
        fill(1e-6, 2);
        fill(dt, model.time_optimal ? 1 : 0);
    ]) for k = 1:N_-1]
    ts = V(zeros(N_))
    ts[1] = t0
    for k = 1:N_-1
        ts[k + 1] = ts[k] + dt
        Altro.discrete_dynamics!(X0[k + 1], EXP, model, X0[k], U0[k], ts[k], dt)
    end

    # bound constraints
    x_max_amid = fill(Inf, n_)
    x_max_abnd = fill(Inf, n_)
    x_max_dt = fill(Inf, n_)
    x_min_amid = fill(-Inf, n_)
    x_min_abnd = fill(-Inf, n_)
    x_min_dt = fill(-Inf, n_)
    u_max_amid = fill(Inf, m_)
    u_max_abnd = fill(Inf, m_)
    u_max_dt = fill(Inf, m_)
    u_min_amid = fill(-Inf, m_)
    u_min_abnd = fill(-Inf, m_)
    u_min_dt = fill(-Inf, m_)
    # constrain the control amplitudes
    x_max_amid[model.controls_idx[1:2]] .= MAX_AMP_NORM_TRANSMON
    x_max_amid[model.controls_idx[3:4]] .= MAX_AMP_NORM_CAVITY
    x_min_amid[model.controls_idx[1:2]] .= -MAX_AMP_NORM_TRANSMON
    x_min_amid[model.controls_idx[3:4]] .= -MAX_AMP_NORM_CAVITY
    # control amplitudes go to zero at boundary
    x_max_abnd[model.controls_idx] .= 0
    x_min_abnd[model.controls_idx] .= 0
    # bound the time step
    if time_optimal
        u_max_dt[model.dt_idx] .= sqrt(dt * 2)
        u_min_dt[model.dt_idx] .= sqrt(dt * 5e-1)
    end
    # vectorize
    x_max_amid = V(x_max_amid)
    x_max_abnd = V(x_max_abnd)
    x_max_dt = V(x_max_dt)
    x_min_amid = V(x_min_amid)
    x_min_abnd = V(x_min_abnd)
    x_min_dt = V(x_min_dt)
    u_max_amid = V(u_max_amid)
    u_max_abnd = V(u_max_abnd)
    u_max_dt = V(u_max_dt)
    u_min_amid = V(u_min_amid)
    u_min_abnd = V(u_min_abnd)
    u_min_dt = V(u_min_dt)

    # constraints
    constraints = Altro.ConstraintList(n_, m_, N_, M, V)
    bc_amid = Altro.BoundConstraint(x_max_amid, x_min_amid, u_max_amid, u_min_amid, n_, m_, M, V)
    Altro.add_constraint!(constraints, bc_amid, V(2:N_-2))
    bc_abnd = Altro.BoundConstraint(x_max_abnd, x_min_abnd, u_max_abnd, u_min_abnd, n_, m_, M, V)
    Altro.add_constraint!(constraints, bc_abnd, V(N_-1:N_-1))
    nbc_cnps = [
        NormBoundConstraint(STATE, cnp_idxs, cnp_tol, n_, m_, M, V) for cnp_idxs in [
            state_idxs(c_level, t_level) for (c_level, t_level) in
            product(cnp_levels, range(0; length=TRANSMON_STATE_COUNT))
        ]
    ]
    for nbc_cnp in nbc_cnps
        Altro.add_constraint!(constraints, nbc_cnp, V(2:N_-1))
    end
    if time_optimal
        bc_dt = Altro.BoundConstraint(x_max_dt, x_min_dt, u_max_dt, u_min_dt, n_, m_, M, V)
        Altro.add_constraint!(constraints, bc_dt, V(1:N_-1))        
    end
    goal_idxs = [model.state1_idx;]
    gc_f = Altro.GoalConstraint(xf, goal_idxs, n_, m_, M, V)
    Altro.add_constraint!(constraints, gc_f, V(N_:N_))

    # cost function
    Q = zeros(n_)
    Q[model.state1_idx] .= qs[1]
    qss = qs[1]
    for level = (target_level + 1):(CAVITY_STATE_COUNT - 1)
        qss *= 2
        level_idxs = cavity_idxs(level)
        Q[level_idxs] .+= qss
    end
    Q[model.controls_idx] .= qs[3]
    Q[model.dcontrols_idx] .= qs[4]
    if model.derivative_count == 1
        Q[model.dstate1_idx] .= qs[5]
    end
    Q = Diagonal(V(Q))
    Qf = Q * N_
    R = V(zeros(m_))
    R[model.d2controls_idx] .= qs[6]
    if model.time_optimal
        R[model.dt_idx] .= qs[7]
    end
    R = Diagonal(R)
    objective = LQRObjective(Q, Qf, R, xf, n_, m_, N_, M, V)
    
    # build problem
    prob = Problem(EXP, model, objective, constraints, X0, U0, ts, N_, M, Md, V)
    # options
    verbose_pn = verbose ? true : false
    verbose_ = verbose ? 2 : 0
    ilqr_max_iterations = smoke_test ? 1 : ilqr_max_iterations
    al_max_iterations = smoke_test ? 1 : al_max_iterations
    n_steps = smoke_test ? 1 : pn_steps
    opts = SolverOptions(
        verbose_pn=verbose_pn, verbose=verbose_,
        projected_newton=true, ilqr_max_iterations=ilqr_max_iterations,
        al_max_iterations=al_max_iterations,
        iterations=max_iterations,
        max_cost_value=max_cost, ilqr_ctol=ilqr_ctol, ilqr_gtol=ilqr_gtol,
        max_state_value=max_state_value,
        max_control_value=max_control_value, penalty_max=max_penalty,
        al_vtol=al_vtol
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
        "save_type" => Int(jl),
        "ilqr_ctol" => ilqr_ctol,
        "ilqr_gtol" => ilqr_gtol,
        "iterations" => iterations_,
        "max_iterations" => max_iterations,
        "pn_steps" => pn_steps,
        "max_cost" => max_cost,
        "derivative_count" => derivative_count,
        "target_level" => target_level,
        "transmon_state_count" => TRANSMON_STATE_COUNT,
        "cavity_state_count" => CAVITY_STATE_COUNT,
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

function state_idxs(c_level, t_level)
    idx = c_level * TRANSMON_STATE_COUNT + t_level + 1
    return [idx; idx + HDIM]
end

function test_model(;derivative_count=0, time_optimal=false)
    Hs = [M(H) for H in (NEGI_H0ROT_ISO, NEGI_H1R_ISO, NEGI_H1I_ISO, NEGI_H2R_ISO, NEGI_H2I_ISO,
                         NEGI_DH0_ISO)]
    model = Model(M, Md, V, Hs, derivative_count, time_optimal)
    return model
end

function test_dynamics()
    # problem
    derivative_count = 1
    time_optimal = true
    Hs = [M(H) for H in (NEGI_H0ROT_ISO, NEGI_H1R_ISO, NEGI_H1I_ISO, NEGI_H2R_ISO, NEGI_H2I_ISO,
                         NEGI_DH0_ISO)]
    model = Model(M, Md, V, Hs, derivative_count, time_optimal)
    n, m = size(model)
    # state + control
    ix = 1:n
    iu = (1:m) .+ n
    x0 = ones(n)
    u0 = ones(m)
    z0 = [x0; u0]
    dt = 2.
    t = 1.
    # tmp
    AB = zeros(n, n + m)
    ABp = zeros(n, n + m)
    Ap = zeros(n, n)
    Bp = zeros(n, m)
    x_tmp = zeros(n)
    # dynamics
    Altro.discrete_dynamics!(x_tmp, EXP, model, x0, u0, t, dt)
    x = Altro.discrete_dynamics(EXP, model, x0, u0, t, dt)
    @assert x ≈ x_tmp
    # jacobian
    # execute autodiff
    f(z) = Altro.discrete_dynamics(EXP, model, z[ix], z[iu], t, dt)
    FD.jacobian!(AB, f, z0)
    A = AB[ix, ix]
    B = AB[ix, iu]
    # execute hand
    Altro.discrete_jacobian!(ABp, Ap, Bp, EXP, model, x0, u0, t, dt, ix, iu)
    # check
    @assert A ≈ Ap
    @assert B ≈ Bp
end


"""
use the result of run_traj to generate data for plotting
"""
function gen_dparam(save_file_path; trial_count=1000, sigma_max=1e-4, save=true)
    # grab relevant information
    (evolution_time, dt, derivative_count, U_, target_level
     ) = h5open(save_file_path, "r") do save_file
        evolution_time = read(save_file, "evolution_time")
        dt = read(save_file, "dt")
        derivative_count = read(save_file, "derivative_count")
        U_ = read(save_file, "acontrols")
        target_level = read(save_file, "target_level")
        return (evolution_time, dt, derivative_count, U_, target_level)
    end
    # set up problem
    n = size(NEGI_H0ROT_ISO, 1)
    mh1 = zeros(n, n) .= NEGI_H0ROT_ISO
    Hs = [M(H) for H in (mh1, NEGI_H1R_ISO, NEGI_H1I_ISO, NEGI_H2R_ISO, NEGI_H2I_ISO,
                         NEGI_DH0_ISO)]
    model = Model(M, Md, V, Hs, derivative_count)
    n, m = size(model)
    N = Int(floor(evolution_time / dt)) + 1
    U = [U_[k, :] for k = 1:N-1]
    X = [zeros(n) for i = 1:N]
    ts = [dt * (k - 1) for k = 1:N]
    # initial state
    x0 = zeros(n)
    x0[model.state1_idx] = IS1_ISO
    x0 = V(x0)
    X[1] = x0
    # target state
    xf = zeros(n)
    cavity_state_ = cavity_state(target_level)
    ψT = kron(cavity_state_, TRANSMON_G)
    xf[model.state1_idx] = get_vec_iso(ψT)
    xf = V(xf)
    # generate parameters
    fq_dev_max = TRANSMON_FREQ * sigma_max
    devs = Array(range(-fq_dev_max, stop=fq_dev_max, length=2 * trial_count + 1))
    fracs = map(d -> d / TRANSMON_FREQ, devs)
    negi_h0rot_iso(dev) = get_mat_iso(
        - 1im * (
            dev * kron(CAVITY_ID, TRANSMON_E * TRANSMON_E')
            + CHI_E_2 * kron(CAVITY_NUMBER, TRANSMON_E * TRANSMON_E')
            + (KAPPA_2 / 2) * kron(CAVITY_QUAD, TRANSMON_ID)
        )
    )
    negi_h0s = map(negi_h0rot_iso, devs)
    gate_errors = zeros(2 * trial_count + 1)
    # collect gate errors
    for (i, negi_h0) in enumerate(negi_h0s)
        mh1 .= negi_h0
        # rollout
        for k = 1:N-1
            Altro.discrete_dynamics!(X[k + 1], EXP, model, X[k], U[k], ts[k], dt)
        end
        ψN = get_vec_uniso(X[N][model.state1_idx])
        gate_error = 1 - abs(ψT'ψN)^2
        gate_errors[i] = gate_error
        # println("dev: $(devs[i]), ge: $(gate_errors[i])\n$(model.Hs[1])")
    end
    # save
    data_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
    if save
        h5open(data_file_path, "w") do data_file
            write(data_file, "save_file_path", save_file_path)
            write(data_file, "gate_errors", gate_errors)
            write(data_file, "devs", devs)
            write(data_file, "fracs", fracs)
        end
    end
    
    return data_file_path
end


function plot_dparam(data_file_paths; labels=nothing, legend=nothing)
    # grab
    gate_errors = []
    fracs = []
    for data_file_path in data_file_paths
        (gate_errors_, fracs_) = h5open(data_file_path, "r") do data_file
            gate_errors_ = read(data_file, "gate_errors")
            fracs_ = read(data_file, "fracs")
            return (gate_errors_, fracs_)
        end
        push!(gate_errors, gate_errors_)
        push!(fracs, fracs_)
    end
    # initial plot
    ytick_vals = Array(-9:1:-1)
    ytick_labels = ["1e$(pow)" for pow in ytick_vals]
    yticks = (ytick_vals, ytick_labels)
    fig = Plots.plot(dpi=DPI, title=nothing, legend=legend, yticks=yticks)
    Plots.xlabel!("|δω_q/ω_q| (1e-4)")
    Plots.ylabel!("Gate Error")
    for i = 1:length(gate_errors)
        gate_errors_ = gate_errors[i]
        fracs_ = fracs[i]
        trial_count = Int((length(fracs_) - 1)/2)
        gate_errors__ = zeros(trial_count + 1)
        # average
        mid = trial_count + 1
        fracs__ = fracs_[mid:end] .* 1e4
        gate_errors__[1] = gate_errors_[mid]
        for j = 1:trial_count
            gate_errors__[j + 1] = (gate_errors_[mid - j] + gate_errors_[mid + j]) / 2
        end
        log_ge = map(x -> log10(x), gate_errors__)
        label = isnothing(labels) ? nothing : labels[i]
        Plots.plot!(fracs__, log_ge, label=label)
    end
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    Plots.savefig(fig, plot_file_path)
    return plot_file_path
end

function plot_population(save_file_path; title="", xlabel="Time (ns)", ylabel="Population")
    # grab
    save_file = read_save(save_file_path)
    transmon_state_count = save_file["transmon_state_count"]
    cavity_state_count = save_file["cavity_state_count"]
    hdim_iso = save_file["hdim_iso"]
    ts = save_file["ts"]
    astates = save_file["astates"]
    N = size(astates, 1)
    d = Int(hdim_iso/2)
    state1_idx = Array(1:hdim_iso)
    # make labels
    transmon_labels = ["g", "e", "f", "h"][1:transmon_state_count]
    cavity_labels = ["$(i)" for i = 0:(cavity_state_count - 1)]
    labels = []
    for p in product(transmon_labels, cavity_labels)
        push!(labels, p[2] * p[1])
    end
    # plot
    fig = Plots.plot(dpi=DPI, title=title, xlabel=xlabel, ylabel=ylabel)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    pops = zeros(N, d)
    for k = 1:N
        ψ = get_vec_uniso(astates[k, state1_idx])
        pops[k, :] = map(x -> abs(x)^2, ψ)
    end
    for i = 1:d
        label = labels[i]
        Plots.plot!(ts, pops[:, i], label=label)
    end
    Plots.savefig(fig, plot_file_path)
    return plot_file_path
end

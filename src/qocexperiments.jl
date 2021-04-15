"""
qocexperiments.jl - A file for common things shared in this repository.
"""

# paths / venv
WDIR = joinpath(@__DIR__, "../")
import Pkg
Pkg.activate(WDIR)

# imports
import Base
using CUDA
using HDF5
using LinearAlgebra
using Plots
using Printf
using RobotDynamics

# plotting configuration and constants
ENV["GKSwstype"] = "nul"
Plots.gr()
const DPI = 300
const MS_SMALL = 2
const MS_MED = 6
const ALPHA = 0.2

# types
@enum SaveType begin
    jl = 1
    samplejl = 2
    py = 3
end


@enum SolverType begin
    ilqr = 1
    alilqr = 2
    altro = 3
end

@enum ArrayType begin
    cpu = 1
    gpu = 2
end


# methods
function generate_file_path(extension, file_name, path)
    # Ensure the path exists.
    mkpath(path)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for (_, _, files) in walkdir(path)
        for file_name_ in files
            if occursin("_$(file_name).$(extension)", file_name_)
                numeric_prefix = parse(Int, split(file_name_, "_")[1])
                max_numeric_prefix = max(numeric_prefix, max_numeric_prefix)
            end
        end
    end
    
    file_path = joinpath(path, "$(lpad(max_numeric_prefix + 1, 5, '0'))_$(file_name).$(extension)")
    return file_path
end


function latest_file_path(extension, file_name, path)
    max_numeric_prefix = -1
    for (_, _, files) in walkdir(path)
        for file_name_ in files
            if occursin("_$(file_name).$(extension)", file_name_)
                numeric_prefix = parse(Int, split(file_name_, "_")[1])
                max_numeric_prefix = max(numeric_prefix, max_numeric_prefix)
            end
        end
    end
    if max_numeric_prefix == -1
        file_path = nothing
    else
        file_path = joinpath(path, "$(lpad(max_numeric_prefix, 5, '0'))_$(file_name).$(extension)")
    end
    return file_path
end


"""
grab_controls - do some extraction of relevant controls
data for common h5 save formats
"""
function grab_controls(save_file_path)
    data = h5open(save_file_path, "r") do save_file
        save_type = SaveType(read(save_file, "save_type"))
        if save_type == jl
            cidx = read(save_file, "controls_idx")
            controls = read(save_file, "astates")[1:end - 1, cidx]
            evolution_time = read(save_file, "evolution_time")
            controls_dt_inv = haskey(save_file, "dt") ? read(save_file, "dt")^(-1) : DT_PREF_INV
        elseif save_type == samplejl
            controls = read(save_file, "controls_sample")[1:end - 1]
            evolution_time = read(save_file, "evolution_time_sample")
            controls_dt_inv = DT_PREF_INV
        elseif save_type == py
            controls = permutedims(read(save_file, "controls"), (2, 1))
            evolution_time = read(save_file, "evolution_time")
            controls_dt_inv = DT_PREF_INV
        end
        return (controls, controls_dt_inv, evolution_time)
    end

    return data
end


"""
read_save - Read all data from an h5 file into memory.
"""
function read_save(save_file_path)
    dict = h5open(save_file_path, "r") do save_file
        dict = Dict()
        for key in keys(save_file)
            dict[key] = read(save_file, key)
        end
        return dict
    end

    return dict
end


function plot_controls(save_file_paths, plot_file_path;
                       labels=nothing,
                       title="", colors=nothing, print_out=true,
                       legend=nothing, d2pi=false)
    fig = Plots.plot(dpi=DPI, title=title, legend=legend)
    for (i, save_file_path) in enumerate(save_file_paths)
        # Grab and prep data.
        (controls, controls_dt_inv, evolution_time) = grab_controls(save_file_path)
        (control_eval_count, control_count) = size(controls)
        control_eval_times = Array(0:control_eval_count - 1) / controls_dt_inv
        if d2pi
            controls = controls ./ (2 * pi)
        end
        
        # Plot.
        for j = 1:control_count
            label = isnothing(labels) ? nothing : labels[i][j]
            color = isnothing(colors) ? :auto : colors[i][j]
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


@inline show_nice(x) = show(IOContext(stdout), "text/plain", x)


@inline get_vec_iso(vec) = vcat(real(vec), imag(vec))


function get_vec_uniso(vec)
    n = size(vec)[1]
    nby2 = Integer(n/2)
    i1 = 1:nby2
    i2 = (nby2 + 1):n
    return vec[i1] + 1im * vec[i2]
end


function get_mat_iso(mat)
    mat_r = real(mat)
    mat_i = imag(mat)
    return vcat(hcat(mat_r, -mat_i),
                hcat(mat_i,  mat_r))
end


function get_mat_uniso(mat)
    n = size(mat, 1)
    nby2 = Integer(n/2)
    i1 = 1:nby2
    i2 = (nby2 + 1):n
    return mat[i1, i1] + mat[i2, i1]
end


@inline get_vec_viso(mat) = reshape(mat, length(mat))


@inline get_vec_unviso(vec) = reshape(vec, (Int(sqrt(length(vec))), Int(sqrt(length(vec)))))


### MATRIX EXPONENTIAL ###

function getrf!(A::AbstractMatrix)
    (A_LU, A_ipiv, info) = LAPACK.getrf!(A)
    return A_LU, A_ipiv
end
@inline getrf!(A::CuMatrix) = CUSOLVER.getrf!(A)
@inline getrs!(trans::Char, A::AbstractMatrix, ipiv::Vector,
               B::AbstractMatrix) = LAPACK.getrs!(trans, A, ipiv, B)
@inline getrs!(trans::Char, A::CuMatrix, ipiv::CuVector,
               B::CuMatrix) = CUSOLVER.getrs!(trans, A, ipiv, B)
const L3 = 1.08e-2
const L5 = 2.0e0-1
const L7 = 7.83e-1
const L9 = 1.78e0
const L13 = 4.74e0
"""
Adapted from [0]
[0] https://github.com/JuliaArrays/StaticArrays.jl/blob/master/src/expm.jl
"""
function exp(A::AbstractMatrix)
    b0 = 64764752532480000.; b1 = 32382376266240000.; b2 = 7771770303897600.
    b3 = 1187353796428800.; b4 = 129060195264000.; b5 = 10559470521600.
    b6 = 670442572800.; b7 = 33522128640.; b8 = 1323241920.; b9 = 40840800.
    b10 = 960960.; b11 = 16380.; b12 = 182.; b13 = 1.
    # get A info
    size_ = size(A, 1)
    nA = maximum(sum(abs.(A); dims=1))
    # scale down
    s = log2(nA / L13)
    if s > 0
        si = ceil(Int, s)
        A = A ./ 2^si
    end
    # compute A
    A2 = A * A
    A4 = A2 * A2
    A6 = A4 * A2
    # compute W1, W2, Z1, Z2
    W1 = b13 * A6 + b11 * A4 + b9 * A2
    W2 = b7 * A6 + b5 * A4 + b3 * A2 + I(size_) * b1
    Z1 = b12 * A6 + b10 * A4 + b8 * A2
    Z2 = b6 * A6 + b4 * A4 + b2 * A2 + I(size_) * b0
    # compute U, V, W
    W = A6 * W1 + W2
    V = A6 * Z1 + Z2
    U = A * W
    # compute R, VmU_LU
    VpU = V + U
    VmU = V - U
    R = VmU \ VpU
    if s > 0
        for i = 1:si
            R = R * R
	    end
    end
    return R
end

"""
Adapted from [0]

Params:
mtmp - length 28 vector of matrices like A
mtmp_dense - length 2 vector of dense matrices like A

Refs:
[0] https://doi.org/10.1137/080716426
"""
function exp!(mtmp::Vector{TM}, mtmp_dense::Vector{TMd}, ipiv_tmp::TVi, A::TM) where {
    TVi<:AbstractVector, TM<:AbstractMatrix, TMd<:AbstractMatrix}
    b0 = 64764752532480000.; b1 = 32382376266240000.; b2 = 7771770303897600.
    b3 = 1187353796428800.; b4 = 129060195264000.; b5 = 10559470521600.
    b6 = 670442572800.; b7 = 33522128640.; b8 = 1323241920.; b9 = 40840800.
    b10 = 960960.; b11 = 16380.; b12 = 182.; b13 = 1.
    # get A info
    size_ = size(A, 1)
    nA = maximum(sum(abs.(A); dims=1))
    # scale down
    A_ = mtmp[1] .= A
    s = log2(nA / L13)
    if s > 0
        si = ceil(Int, s)
        lmul!(1 / 2^si, A_)
    end
    # A - mtmp1
    # compute A
    A2 = mul!(mtmp[3], A_, A_)
    A4 = mul!(mtmp[4], A2, A2)
    A6 = mul!(mtmp[5], A4, A2)
    # A2 - mtmp3
    # A4 - mtmp4
    # A6 - mtmp5
    # compute W1, W2, Z1, Z2
    W1 = mtmp[6]
    W2 = mtmp[7]
    Z1 = mtmp[8]
    Z2 = mtmp[9]
    for i in eachindex(A)
        W1[i] = b13 * A6[i] + b11 * A4[i] + b9 * A2[i]
        W2[i] = b7 * A6[i] + b5 * A4[i] + b3 * A2[i]
        Z1[i] = b12 * A6[i] + b10 * A4[i] + b8 * A2[i]
        Z2[i] = b6 * A6[i] + b4 * A4[i] + b2 * A2[i]
    end
    for i = 1:size_
        W2[i, i] += b1
        Z2[i, i] += b0
    end
    # W1 - mtmp6
    # W2 - mtmp7
    # Z1 - mtmp8
    # Z2 - mtmp9
    # compute U, V, W
    U = mtmp[10]
    V = mtmp[11]
    W = mtmp[12]
    mul!(W, A6, W1)
    mul!(V, A6, Z1)
    for i in eachindex(A)
        W[i] += W2[i]
        V[i] += Z2[i]
    end
    mul!(U, A_, W)
    # U - mtmp10
    # V - mtmp11
    # W - mtmp12
    # compute R_unscaled, VmU_LU
    VmU = mtmp_dense[1]
    VpU = mtmp_dense[2]
    for i in eachindex(A)
        VpU[i] = V[i] + U[i]
        VmU[i] = V[i] - U[i]
    end
    (VmU_LU, VmU_ipiv) = getrf!(VmU)
    ipiv_tmp .= VmU_ipiv
    mtmp[13] .= getrs!('N', VmU_LU, VmU_ipiv, VpU)
    # R_unscaled - mtmp13 - mtmp_dense2
    # VmU_LU - mtmp_dense1
    # VmU_ipiv - ipiv_tmp
    # compute R_scaled
    R = mtmp[14] .= mtmp[13]
    if s > 0
        for t = 1:si
            mul!(mtmp[15], R, R)
            R .= mtmp[15]
        end
    end
    # R_scaled - mtmp14
    return R
end


"""
Adapted from [0]

Params:
mtmp - length 28 vector of matrices like A
mtmp_dense - length 2 vector of dense matrices like A

Refs:
[0] https://github.com/scipy/scipy/blob/master/scipy/linalg/_expm_frechet.py#L223
"""
function exp_frechet!(mtmp::Vector{TM}, mtmp_dense::Vector{TMd},
                      ipiv_tmp::TVi, A::TM, E::TM; reuse::Bool=false) where {
    TVi<:AbstractVector,TM<:AbstractMatrix,TMd<:AbstractMatrix}
    b0 = 64764752532480000.; b1 = 32382376266240000.; b2 = 7771770303897600.
    b3 = 1187353796428800.; b4 = 129060195264000.; b5 = 10559470521600.
    b6 = 670442572800.; b7 = 33522128640.; b8 = 1323241920.; b9 = 40840800.
    b10 = 960960.; b11 = 16380.; b12 = 182.; b13 = 1.
    # get A info
    size_ = size(A, 1)
    nA = maximum(sum(abs.(A); dims=1))
    # scale down
    E_ = mtmp[2] .= E
    s = log2(nA / L13)
    if s > 0
        si = ceil(Int, s)
        lmul!(1 / 2^si, E_)
    end
    # E - mtmp2
    # compute A, U, V, etc.
    if !reuse
        exp!(mtmp, mtmp_dense, ipiv_tmp, A)
    end
    A_ = mtmp[1]
    A2 = mtmp[3]
    A4 = mtmp[4]
    A6 = mtmp[5]
    W1 = mtmp[6]
    W2 = mtmp[7]
    Z1 = mtmp[8]
    Z2 = mtmp[9]
    U = mtmp[10]
    V = mtmp[11]
    W = mtmp[12]
    R = mtmp[13]
    VmU_LU = mtmp_dense[1]
    VmU_ipiv = ipiv_tmp
    # compute M
    M2 = mul!(mtmp[15], A_, E_)
    mul!(mtmp[16], E_, A_)
    for i in eachindex(A)
        M2[i] += mtmp[16][i]
    end
    M4 = mul!(mtmp[16], M2, A2)
    mul!(mtmp[17], A2, M2)
    for i in eachindex(A)
        M4[i] += mtmp[17][i]
    end
    M6 = mul!(mtmp[17], A4, M2)
    mul!(mtmp[18], M4, A2)
    # M2 - mtmp15
    # M4 - mtmp16
    # M6 - mtmp17
    # compute Lw1, Lw2, Lz1, Lz2
    Lw1 = mtmp[18]
    Lw2 = mtmp[19]
    Lz1 = mtmp[20]
    Lz2 = mtmp[21]
    for i in eachindex(A)
        M6[i] += mtmp[18][i]
        Lw1[i] = b13 * M6[i] + b11 * M4[i] + b9 * M2[i]
        Lw2[i] = b7 * M6[i] + b5 * M4[i] + b3 * M2[i]
        Lz1[i] = b12 * M6[i] + b10 * M4[i] + b8 * M2[i]
        Lz2[i] = b6 * M6[i] + b4 * M4[i] + b2 * M2[i]
    end
    # Lw1 - mtmp18
    # Lw2 - mtmp19
    # Lz1 - mtmp20
    # Lz2 - mtmp21
    # Compute Lw, Lu, Lv
    Lw = mul!(mtmp[22], A6, Lw1)
    mul!(mtmp[23], M6, W1)
    for i in eachindex(A)
        Lw[i] += mtmp[23][i] + Lw2[i]
    end
    Lu = mul!(mtmp[23], A_, Lw)
    mul!(mtmp[25], E_, W)
    Lv = mul!(mtmp[24], A6, Lz1)
    mul!(mtmp[26], M6, Z1)
    for i in eachindex(A)
        Lu[i] += mtmp[25][i]
        Lv[i] += mtmp[26][i] + Lz2[i]
    end
    # Lw - mtmp 22
    # Lu - mtmp 23
    # Lv - mtmp 24
    # compute L
    for i in eachindex(A)
        mtmp[26][i] = Lu[i] - Lv[i]
    end
    L = mul!(mtmp[25], mtmp[26], R)
    for i in eachindex(A)
        L[i] += Lu[i] + Lv[i]
    end
    mtmp_dense[2] .= L
    L .= getrs!('N', VmU_LU, VmU_ipiv, mtmp_dense[2])
    # don't overwrite mtmp[13] so that multiple exp_frechet!
    # calls can reuse mtmp[13] for R_unscaled
    R = mtmp[15] .= mtmp[13]
    # scale up
    if s > 0
        for t = 1:si
            mul!(mtmp[26], R, L)
            mul!(mtmp[27], L, R)
            mul!(mtmp[28], R, R)
            for i in eachindex(A)
                L[i] = mtmp[26][i] + mtmp[27][i]
                R[i] = mtmp[28][i]
            end
        end
    end
    # L - mtmp25
    # R - mtmp15
    return L
end

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

# Adapted from implementation in StaticArrays
# https://github.com/JuliaArrays/StaticArrays.jl/blob/master/src/expm.jl
# Adapted from implementation in Base; algorithm from
# Higham, "Functions of Matrices: Theory and Computation", SIAM, 2008
function exp_(_A::AbstractMatrix{T}) where T
    S = typeof((zero(T)*zero(T) + zero(T)*zero(T))/one(T))
    A = _A # S.(_A)
    # omitted: matrix balancing, i.e., LAPACK.gebal!
    nA = maximum(sum(abs.(A); dims=1))    # marginally more performant than norm(A, 1)
    ## For sufficiently small nA, use lower order Padé-Approximations
    if (nA <= 2.1)
        A2 = A*A
        if nA > 0.95
            U = @evalpoly(A2, S(8821612800)*I, S(302702400)*I, S(2162160)*I, S(3960)*I, S(1)*I)
            U = A*U
            V = @evalpoly(A2, S(17643225600)*I, S(2075673600)*I, S(30270240)*I, S(110880)*I, S(90)*I)
        elseif nA > 0.25
            U = @evalpoly(A2, S(8648640)*I, S(277200)*I, S(1512)*I, S(1)*I)
            U = A*U
            V = @evalpoly(A2, S(17297280)*I, S(1995840)*I, S(25200)*I, S(56)*I)
        elseif nA > 0.015
            U = @evalpoly(A2, S(15120)*I, S(420)*I, S(1)*I)
            U = A*U
            V = @evalpoly(A2, S(30240)*I, S(3360)*I, S(30)*I)
        else
            U = @evalpoly(A2, S(60)*I, S(1)*I)
            U = A*U
            V = @evalpoly(A2, S(120)*I, S(12)*I)
        end
        expA = (V - U) \ (V + U)
    else
        s  = log2(nA/5.4)               # power of 2 later reversed by squaring
        if s > 0
            si = ceil(Int,s)
            A = A / S(2^si)
        end

        A2 = A*A
        A4 = A2*A2
        A6 = A2*A4

        U = A6*(S(1)*A6 + S(16380)*A4 + S(40840800)*A2) +
            (S(33522128640)*A6 + S(10559470521600)*A4 + S(1187353796428800)*A2) +
            S(32382376266240000)*I
        U = A*U
        V = A6*(S(182)*A6 + S(960960)*A4 + S(1323241920)*A2) +
            (S(670442572800)*A6 + S(129060195264000)*A4 + S(7771770303897600)*A2) +
            S(64764752532480000)*I
        expA = (V - U) \ (V + U)

        if s > 0            # squaring to reverse dividing by power of 2
            for t=1:si
                expA = expA*expA
            end
        end
    end

    expA
end

const L3 = 1.08e-2
const L5 = 2.0e0-1
const L7 = 7.83e-1
const L9 = 1.78e0
const L13 = 4.74e0
function exp_frechet!(tmp::Vector{AbstractMatrix}, A::AbstractMatrix, E::AbstractMatrix;
                      reuse_UV::Bool=false)
    size_ = size(A, 1)
    nA = maximum(sum(abs.(A); dims=1))
    if (nA <= L9)
        if nA <= L3
            b0 = 120.; b1 = 60.; b2 = 12.; b3 = 1.
            if !reuse_UV
                # compute A
                A2 = mul!(tmp[1], A, A)
                # A2 - tmp 1
                # compute U, V
                V = tmp[3]
                for i in eachindex(A)
                    tmp[4][i] = b3 * A2[i]
                    V = b2 * A2[i]
                end
                for i = 1:size_
                    tmp[4][i, i] += b1
                    V[i, i] += b0
                end
                U = mul!(tmp[2], A, tmp[4])
                # U - tmp2
                # V - tmp3
            else
                A2 = tmp[1]
                U = tmp[2]
                V = tmp[3]
            end
            # compute M
            M2 = mul!(tmp[4], A, E)
            mul!(tmp[5], E, A)
            # M2 - tmp4
            for i in eachindex(A)
                M2[i] += tmp[5][i]
                tmp[5][i] = b3 * M2[i]
                tmp[6][i] = b3 * A2[i]
            end
            for i = 1:size_
                tmp[6][i, i] += b1
            end
            mul!(tmp[7], A, tmp[5])
            mul!(tmp[8], E, tmp[6])
            Lu = tmp[5]
            Lv = tmp[6]
            VmU = tmp[7]
            VpU = tmp[8]
            for i in eachindex(A)
                Lu[i] = tmp[7][i] + tmp[8][i]
                Lv[i] = b2 * M2[i]
                VmU[i] = V[i] - U[i]
                VpU[i] = V[i] + U[i]
            end
            # Lu - tmp5
            # Lv - tmp6
            # VmU - tmp7
            # VpU - tmp8
            # compute R
            (R, VmU_LU, VmU_ipiv) = gesv!(VmU, VpU)
            # VmU_LU - tmp7
            # R - tmp8
            # compute L
            for i in eachindex(A)
                tmp[10][i] = Lu[i] - Lv[i]
            end
            L = mul!(tmp[9], tmp[10], R)
            for i in eachindex(A)
                L[i] += Lu[i] + Lv[i]
            end
            getrs!("N", VmU_LU, VmU_ipiv, L)
            # L - tmp9
            return L
        elseif nA <= L5
            U = @evalpoly(A2, S(15120)*I, S(420)*I, S(1)*I)
            U = A*U
            V = @evalpoly(A2, S(30240)*I, S(3360)*I, S(30)*I)
        elseif nA <= L7
            b0 = 17297280.; b1 = 8648640.; b2 = 1995840.; b3 = 277200.
            b4 = 25200.; b5 = 1512.; b6 = 56.; b7 = 1.
        else # nA <= L9
            b0 = 17643225600.; b1 = 8821612800.; b2 = 2075673600.; b3 = 302702400.
            b4 = 30270240.; b5 = 2162160.; b6 = 110880.; b7 = 3960.; b8 = 90.; b9 = 1.
            if !reuse_UV
                # compute A
                A2 = mul!(tmp[1], A, A)
                A4 = mul!(tmp[2], A2, A2)
                A6 = mul!(tmp[3], A4, A2)
                A8 = mul!(tmp[4], A4, A4)
                # A2 - tmp1
                # A4 - tmp2
                # A6 - tmp3
                # A8 - tmp4
                # compute U, V
                V = tmp[6]
                for i in eachindex(A)
                    tmp[7][i] = b3 * A2[i] + b4 * A4[i] + b7 * A6[i]
                    V[i] = b2 * A2[i] + b4 * A4[i] + b6 * A6[i] + b8 * A8[i]
                end
                for i = 1:size_
                    tmp[7][i, i] += b1
                    V[i, i] += b0
                end
                U = mul!(tmp[5], A, tmp[7])
                # U - tmp5
                # V - tmp6
            else
                A2 = tmp[1]
                A4 = tmp[2]
                A6 = tmp[3]
                A8 = tmp[4]
                U = tmp[5]
                V = tmp[6]
            end
            # compute Ms
            M2 = mul!(tmp[7], A, E)
            mul!(tmp[8], E, A)
            for i in eachindex(A)
                M2[i] += tmp[8][i]
            end
            M4 = mul!(tmp[8], M2, A2)
            mul!(tmp[9], A2, M2)
            for i in eachindex(A)
                M4[i] += tmp[9][i]
            end
            M6 = mul!(tmp[9], A4, M2)
            mul!(tmp[10], M4, A2)
            for i in eachindex(A)
                M6[i] += tmp[10][i]
            end
            M8 = mul!(tmp[10], A4, M4)
            mul!(tmp[11], M4, A4)
            # M2 - tmp7
            # M4 - tmp8
            # M6 - tmp9
            # M8 - tmp10
            # compute Lu, Lv
            for i in eachindex(A)
                M8[i] += tmp[11][i]
                tmp[11][i] = b9 * M8[i] + b7 * M6[i] + b5 * M4[i] + b3 * M2[i]
                tmp[12][i] = b9 * A8[i] + b7 * A6[i] + b5 * A4[i] + b3 * A2[i]
            end
            for i = 1:size_
                tmp[12][i, i] += b1
            end
            mul!(tmp[13], A, tmp[11])
            mul!(tmp[14], E, tmp[12])
            Lu = tmp[11]
            Lv = tmp[12]
            VmU = tmp[13]
            VpU = tmp[14]
            for i in eachindex(A)
                Lu[i] = tmp[13][i] + tmp[14][i]
                Lv[i] = b8 * M8[i] + b6 * M6[i] + b4 * M4[i] + b2 * M2[i]
                VmU[i] = V[i] - U[i]
                VpU[i] = V[i] + U[i]
            end
            # Lu - tmp11
            # Lv - tmp12
            # VmU - tmp13
            # VpU - tmp14
            # compute R
            (R, VmU_LU, VmU_ipiv) = gesv!(VmU, VpU)
            # VmU_LU - tmp13
            # R - tmp14
            # compute L
            for i in eachindex(A)
                tmp[16][i] = Lu[i] - Lv[i]
            end
            L = mul!(tmp[15], tmp[16], R)
            for i in eachindex(A)
                L[i] += Lu[i] + Lv[i]
            end
            getrs!("N", VmU_LU, VmU_ipiv, L)
            # L - tmp15
            return L
        end
    else
        s = log2(nA/L13)
        if s > 0
            si = ceil(Int, s)
            lmul!(1 / 2^si, A)
        end

        if s > 0
            for t=1:si
                L = R * L + L * R
            end
        end
    end
end


# # Taken from
# # https://github.com/JuliaGPU/CUDA.jl/blob/621972b4cb3e386dbbd43f2cd873325e59c8d60f/lib/cusolver/dense.jl#L106
# function CUDA.CUSOLVER.getrf!(A::CuMatrix)
#     show_nice(A)
#     m,n = size(A)
#     lda = max(1, stride(A, 2))

#     devipiv = CuArray{Cint}(undef, min(m,n))
#     devinfo = CuArray{Cint}(undef, 1)
#     CUDA.@workspace eltyp=Float64 size=CUDA.@argout(
#         CUDA.CUSOLVER.cusolverDnDgetrf_bufferSize(CUDA.CUSOLVER.dense_handle(), m, n, A, lda, out(Ref{Cint}(0)))
#         )[] buffer->begin
#             CUDA.CUSOLVER.cusolverDnDgetrf(CUDA.CUSOLVER.dense_handle(), m, n, A, lda, buffer, devipiv, devinfo)
#         end

#     info = CUDA.@allowscalar devinfo[1]
#     CUDA.unsafe_free!(devinfo)
#     println(devipiv)
#     # if info < 0
#     #     throw(ArgumentError("The $(info)th parameter is wrong"))
#     # elseif info > 0
#     #     throw(LinearAlgebra.SingularException(info))
#     # end

#     A, devipiv
# end


# # Taken from
# # https://github.com/JuliaGPU/CUDA.jl/blob/621972b4cb3e386dbbd43f2cd873325e59c8d60f/lib/cusolver/dense.jl#L197
# function CUDA.CUSOLVER.getrs!(trans::Char,
#                               A::CuMatrix,
#                               ipiv::CuVector{Cint},
#                               B::CuVecOrMat)
#     n = size(A, 1)
#     if size(A, 2) != n
#         throw(DimensionMismatch("LU factored matrix A must be square!"))
#     end
#     if size(B, 1) != n
#         throw(DimensionMismatch("first dimension of B, $(size(B,1)), must match second dimension of A, $n"))
#     end
#     nrhs = size(B, 2)
#     lda  = max(1, stride(A, 2))
#     ldb  = max(1, stride(B, 2))

#     devinfo = CuArray{Cint}(undef, 1)
#     CUDA.CUSOLVER.cusolverDnDgetrs(CUDA.CUSOLVER.dense_handle(), trans, n, nrhs, A, lda, ipiv, B, ldb, devinfo)

#     info = CUDA.@allowscalar devinfo[1]
#     CUDA.unsafe_free!(devinfo)
#     if info < 0
#         throw(ArgumentError("The $(info)th parameter is wrong"))
#     end
#     B
# end

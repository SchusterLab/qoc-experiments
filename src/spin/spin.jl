"""
spin.jl
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "qocexperiments.jl"))

# imports
using LinearAlgebra

# paths
const MM_OUT_PATH = abspath(joinpath(WDIR, "out", "mm"))

# experimental constants
const WQ = 2π * 1.4e-2 #GHz
const A_MAX = 2π * 0.5 #GHz

# simulation constants
const DT_PREF = 1e-1

# define the system
const HDIM = 2
const HDIM_ISO = 2 * HDIM
# operators
const SIGMAX = [0 1;
                1 0]
const SIGMAZ = [1 0;
                0 -1]
# hamiltonian
const NEGI_H0_ISO = get_mat_iso(-1im * WQ * SIGMAZ / 2)
const NEGI_H1_ISO = get_mat_iso(-1im * SIGMAX / 2)

# initial states
const ID = Array{Float64,2}(I(HDIM))
const IS1_ISO = get_vec_iso(ID[:,1])
const IS2_ISO = get_vec_iso(ID[:,2])
# target states
const XPIBY2 = [1 -1im;
                -1im 1] / sqrt(2)
const XPIBY21_ISO = get_vec_iso(XPIBY2[:,1])
const XPIBY22_ISO = get_vec_iso(XPIBY2[:,2])

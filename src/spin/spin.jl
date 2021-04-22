"""
spin.jl
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "qocexperiments.jl"))

# paths
const MM_OUT_PATH = abspath(joinpath(WDIR, "out", "mm"))

# experimental constants
const WQ = 2π * 1.4e-2 #GHz
const A_MAX = 2π * 0.5 #GHz

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
# states
const IS1_ISO = [1., 0, 0, 0]

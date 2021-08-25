"""
mm.jl - multimode
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "qocexperiments.jl"))

# `using Package` is like `from Package import *`in Python.
# `import Package` is like `import Package` in Python.
# `import Package: function` is like `from Package import function` in Python.
# Importing is a one-time cost so importing the
# entire package isn't a big deal unless there are name-space conflicts,
# e.g. two functions have the same name.
using HDF5
using LinearAlgebra
import Plots
using Printf
using StaticArrays

# paths
const MM_OUT_PATH = abspath(joinpath(WDIR, "out", "mm"))

# Simulation constants.
const DT_PREF = 1e-2
const DT_PREF_INV = 100

# Define experimental constants.
const TRANSMON_FREQ = 2π * 4.99
const CHI_E_2 = 2π * -1.33e-3
const CAVITY_FREQ_2 = 2π * 5.96
const KAPPA_2 = 2π * 5.23e-6
const MAX_AMP_NORM_TRANSMON = 2π * 4e-2
const MAX_AMP_NORM_CAVITY = 2π * 4e-3

# Define the system.
const TRANSMON_STATE_COUNT = 2
const CAVITY_STATE_COUNT = 20
const HDIM = TRANSMON_STATE_COUNT * CAVITY_STATE_COUNT
const HDIM_ISO = 2 * HDIM

# This is placing sqrt(1), ..., sqrt(TRANSMON_STATE_COUNT - 1) on the 1st diagonal
# counted up from the true diagonal.
const TRANSMON_ANNIHILATE = diagm(1 => map(sqrt, 1:TRANSMON_STATE_COUNT-1))
# Taking the adjoint is as simple as adding an apostrophe.
const TRANSMON_CREATE = TRANSMON_ANNIHILATE'
# In julia `a * b` is scalar multiplication where a and b are scalars,
# `a * b` is matrix multiplication where a and b are vectors,
# and `a .* b` is the element-wise product where a and b are vectors.
const TRANSMON_NUMBER = TRANSMON_CREATE * TRANSMON_ANNIHILATE
const TRANSMON_ID = I(TRANSMON_STATE_COUNT)
# Julia is 1-indexed, that means the first element in the
# array is `arr[1]` not like `arr[0]` in python.
# To get the last element of the array you do `arr[end]`
# or `arr[end - 1]` which is like `arr[-1]` or `arr[-2]` in Python.
const TRANSMON_G = [1; zeros(TRANSMON_STATE_COUNT - 1)]
const TRANSMON_E = [zeros(1); 1; zeros(TRANSMON_STATE_COUNT - 2)]
function transmon_state(level)
    state = zeros(TRANSMON_STATE_COUNT)
    state[level + 1] = 1.
    return state
end

const CAVITY_ANNIHILATE = diagm(1 => map(sqrt, 1:CAVITY_STATE_COUNT-1))
const CAVITY_CREATE = CAVITY_ANNIHILATE'
const CAVITY_NUMBER = CAVITY_CREATE * CAVITY_ANNIHILATE
const CAVITY_ID = I(CAVITY_STATE_COUNT)
const CAVITY_QUAD = CAVITY_NUMBER * (CAVITY_NUMBER - CAVITY_ID)
const CAVITY_ZERO = [1; zeros(CAVITY_STATE_COUNT - 1)]
const CAVITY_ONE = [zeros(1); 1; zeros(CAVITY_STATE_COUNT - 2)]
# const CAVITY_TWO = [zeros(2); 1; zeros(CAVITY_STATE_COUNT - 3)]
# const CAVITY_THREE = [zeros(3); 1; zeros(CAVITY_STATE_COUNT - 4)]
# const CAVITY_FOUR = [zeros(4); 1; zeros(CAVITY_STATE_COUNT - 5)]
function cavity_state(level)
    state = zeros(CAVITY_STATE_COUNT)
    state[level + 1] = 1.
    return state
end


# Static hamiltonian.
# We define all of our hamiltonians as -1im * H so that when we write
# the schroedinger equation dynamics we don't have to do extra
# multiplication.
const NEGI_H0_ISO = get_mat_iso(
    - 1im * (
        TRANSMON_FREQ * kron(CAVITY_ID, TRANSMON_E * TRANSMON_E')
        + CHI_E_2 * kron(CAVITY_NUMBER, TRANSMON_E * TRANSMON_E')
        + CAVITY_FREQ_2 * kron(CAVITY_NUMBER, TRANSMON_ID)
        + (KAPPA_2 / 2) * kron(CAVITY_QUAD, TRANSMON_ID)
    )
)
const NEGI_H0ROT_ISO = get_mat_iso(
    - 1im * (
        # TRANSMON_FREQ * kron(CAVITY_ID, TRANSMON_E * TRANSMON_E')
        + CHI_E_2 * kron(CAVITY_NUMBER, TRANSMON_E * TRANSMON_E')
        # + CAVITY_FREQ_2 * kron(CAVITY_NUMBER, TRANSMON_ID)
        + (KAPPA_2 / 2) * kron(CAVITY_QUAD, TRANSMON_ID)
    )
)
const NEGI_DH0_ISO = get_mat_iso(
    - 1im * (
        kron(CAVITY_ID, TRANSMON_E * TRANSMON_E')
        # + kron(CAVITY_NUMBER, TRANSMON_E * TRANSMON_E')
        # + kron(CAVITY_NUMBER, TRANSMON_ID)
        # + (kron(CAVITY_QUAD, TRANSMON_ID) ./ 2)
    )
)

# Control hamiltonians.
# Complex automatic differentiation in julia is not fully supported yet,
# so we do the same u(t) a + conj(u(t)) a_dagger where u = ux + i uy
# business.
# Transmon drive.
const NEGI_H1R_ISO = get_mat_iso(
    -1im * kron(CAVITY_ID, TRANSMON_G * TRANSMON_E' + TRANSMON_E * TRANSMON_G')
)
const NEGI_H1I_ISO = get_mat_iso(
    -1im * kron(CAVITY_ID, 1im * (TRANSMON_G * TRANSMON_E' - TRANSMON_E * TRANSMON_G'))
)
# Cavity drive.
const NEGI_H2R_ISO = get_mat_iso(
    -1im * kron(CAVITY_ANNIHILATE + CAVITY_CREATE, TRANSMON_ID)
)
const NEGI_H2I_ISO = get_mat_iso(
    -1im * kron(1im * (CAVITY_ANNIHILATE - CAVITY_CREATE), TRANSMON_ID)
)

# Initial states.
const IS1_ISO = get_vec_iso(
    kron(CAVITY_ZERO, TRANSMON_G)
)

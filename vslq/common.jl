"""
common.jl - A common file for the vslq-spinspin directory.
"""

using HDF5
using LinearAlgebra
using Plots
using Printf
using StaticArrays

### COMMON ###

# simulation constants
DT_PREF = 1e-2
DT_PREF_INV = 1e2

# plotting configuration and constants
ENV["GKSwstype"] = "nul"
Plots.gr()
DPI = 300
MS = 2
ALPHA = 0.2

# methods
"""
generate_save_file_path - generate a save file path like
    "save_path/XXXXX_save_file_name.extension" where X
    is an integer.
"""
function generate_save_file_path(extension, save_file_name, save_path)
    # Ensure the path exists.
    mkpath(save_path)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for (_, _, files) in walkdir(save_path)
        for file_name in files
            if occursin("_$(save_file_name).$(extension)", file_name)
                max_numeric_prefix = max(parse(Int, split(file_name, "_")[1]))
            end
        end
    end

    save_file_name = "_$(save_file_name).$(extension)"
    save_file_name = @sprintf("%05d%s", max_numeric_prefix + 1, save_file_name)

    return joinpath(save_path, save_file_name)
end


"""
read_save - Read all data from an h5 file into memory.
"""
function read_save(save_file_path)
    dict = h5open(save_file_path, "r") do save_file
        dict = Dict()
        for key in names(save_file)
            dict[key] = read(save_file, key)
        end
        return dict
    end

    return dict
end


show_nice(x) = show(IOContext(stdout), "text/plain", x)


function get_vec_iso(vec)
    return [real(vec);
            imag(vec)]
end


function get_mat_iso(mat)
    len = size(mat)[1]
    mat_r = real(mat)
    mat_i = imag(mat)
    return [mat_r -mat_i;
            mat_i  mat_r]
end


### VSLQ SPIN SPIN ###

# Define experimental constants.
OMEGA_1 = 3.5 #GHz
OMEGA_2 = 4.2 #GHz
G = -2.1e-3 #GHz
MAX_AMP_1 = 1.e1 #GHz
MAX_AMP_2 = 1.e1 #GHz

# Define the system.
# ISO means the object is defined in the complex to real isomorphism.
# NEG_I is the negative complex unit
HILBERT_SIZE_ISO = 8
_II_ISO = get_mat_iso(kron(I(2), I(2)))
II_ISO_1 = SVector{HILBERT_SIZE_ISO}(_II_ISO[:, 1])
II_ISO_2 = SVector{HILBERT_SIZE_ISO}(_II_ISO[:, 2])
II_ISO_3 = SVector{HILBERT_SIZE_ISO}(_II_ISO[:, 3])
II_ISO_4 = SVector{HILBERT_SIZE_ISO}(_II_ISO[:, 4])
XPIBY2 = [1 -1im;
          -1im 1] ./ sqrt(2)
_XPIBY2I_ISO = get_mat_iso(kron(XPIBY2, I(2)))
XPIBY2I_ISO_1 = SVector{HILBERT_SIZE_ISO}(_XPIBY2I_ISO[:, 1])
XPIBY2I_ISO_2 = SVector{HILBERT_SIZE_ISO}(_XPIBY2I_ISO[:, 2])
XPIBY2I_ISO_3 = SVector{HILBERT_SIZE_ISO}(_XPIBY2I_ISO[:, 3])
XPIBY2I_ISO_4 = SVector{HILBERT_SIZE_ISO}(_XPIBY2I_ISO[:, 4])
_IXPIBY2_ISO = get_mat_iso(kron(I(2), XPIBY2))
IXPIBY2_ISO_1 = SVector{HILBERT_SIZE_ISO}(_IXPIBY2_ISO[:, 1])
IXPIBY2_ISO_2 = SVector{HILBERT_SIZE_ISO}(_IXPIBY2_ISO[:, 2])
IXPIBY2_ISO_3 = SVector{HILBERT_SIZE_ISO}(_IXPIBY2_ISO[:, 3])
IXPIBY2_ISO_4 = SVector{HILBERT_SIZE_ISO}(_IXPIBY2_ISO[:, 4])

# SIGMA_X, SIGMA_Z are the X and Z Pauli matrices.
SIGMA_X = [0 1;
           1 0]
SIGMA_Z = [1 0;
           0 -1]
NEGI_ZI_ISO = SMatrix{HILBERT_SIZE_ISO, HILBERT_SIZE_ISO}(
    get_mat_iso(kron(-1im * SIGMA_Z, I(2)))
)
NEGI_IZ_ISO = SMatrix{HILBERT_SIZE_ISO, HILBERT_SIZE_ISO}(
    get_mat_iso(kron(-1im * I(2), SIGMA_Z))
)
NEGI_ZZ_ISO = SMatrix{HILBERT_SIZE_ISO, HILBERT_SIZE_ISO}(
    get_mat_iso(kron(-1im * SIGMA_Z, SIGMA_Z))
)
NEGI_XI_ISO = SMatrix{HILBERT_SIZE_ISO, HILBERT_SIZE_ISO}(
    get_mat_iso(kron(-1im * SIGMA_X, I(2)))
)
NEGI_IX_ISO = SMatrix{HILBERT_SIZE_ISO, HILBERT_SIZE_ISO}(
    get_mat_iso(kron(-1im * I(2), SIGMA_X))
)
NEGI_H0_ISO = 2 * pi * (
    -(OMEGA_1 / 2 + G) * NEGI_ZI_ISO
    -(OMEGA_2 / 2 + G) * NEGI_IZ_ISO
    + G * NEGI_ZZ_ISO
)
NEGI_H1_ISO = 2 * pi * NEGI_XI_ISO
NEGI_H2_ISO = 2 * pi * NEGI_IX_ISO

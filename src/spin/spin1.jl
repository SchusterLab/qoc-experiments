"""
spin1.jl
"""

"""
mm1.jl - first multimode experiment
"""

WDIR = joinpath(@__DIR__, "../..")
include(joinpath(WDIR, "src", "spin", "spin.jl"))

using Altro
using ForwardDiff
using HDF5
using LinearAlgebra
using RobotDynamics
using TrajectoryOptimization
const RD = RobotDynamics
const TO = TrajectoryOptimization

const



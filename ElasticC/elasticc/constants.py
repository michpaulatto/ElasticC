"""
Author: michpaulatto
GitHub: ElasticC
"""

# Various constants and variables
import math

# Constants
RR = 8.314472	# Gas constant J/mol

# Variables
FF = 1.0	# Dependence of Q on frequency
HH = 2.76e5	# Activation enthalpy J/mol
dvpdt0 = -5.7e-1
bbvp = -8.1e-5	# K-1 anharmonic dlnVp/Dt from Dunn 2000, Christensen 1979
const = FF*HH/math.pi/RR

# Parameters for dependence of Q on temperature
alpha = 0.15
f = 10.0
d = 500.


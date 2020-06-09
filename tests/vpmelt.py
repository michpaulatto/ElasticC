import os
import sys
sys.path.append(os.path.abspath('..'))

import elasticc as el

import numpy as np
import matplotlib.pyplot as plt

# Almost zero. This is needed to avoid some division by zero.
# Need a better solution
zero = 1e-12

# Set up components using values from Mainprice (1997)
# Solid phase properties
ro1 = 2.84     # kg/m^3
c11 = 136.03   # GPa
c12 = 56.31    # GPa
c44 = 39.86    # GPa
K1 = (c11+2*c12)/3.0
G1 = c44

# Melt phase properties
ro2 = 2.7     # Kg/m^3
K2 = 1.491e1   # GPa
G2 = zero      # GPa

# Seismic velocities
Vp1 = np.sqrt((K1+4.0/3.0*G1)/ro1)
Vs1 = np.sqrt(G1/ro1)
Vp2 = np.sqrt((K2+4.0/3.0*G2)/ro2)
Vs2 = np.sqrt(G2/ro2)

print(dir(el))
# Initialize volume fractions (melt fraction from 0 to 1)
dc = 5e-3
nc = int(1./dc-1)
c2 = np.linspace(dc,dc*nc,nc)
c1 = 1-c2

# Calculate density of composite as function of melt fraction
roe = el.limits.Voigt(ro1,c1,ro2,c2)

###############################################

# Reuss bounds
Kr = el.limits.Reuss(K1,c1,K2,c2)
Gr = el.limits.Reuss(K1,c1,K2,c2)
Vpr = np.sqrt((Kr+4.0/3.0*Gr)/roe)
Vsr = np.sqrt(Gr/roe)

# Voight
Kv = el.limits.Voigt(K1,c1,K2,c2)
Gv = el.limits.Voigt(K1,c1,K2,c2)
Vpv = np.sqrt((Kv+4.0/3.0*Gv)/roe)
Vsv = np.sqrt(Gv/roe)

# Hashin-Shtrikman
Khs1,Ghs1 = el.limits.hs_bounds(K1,K2,G1,G2,c1,c2)
Khs2,Ghs2 = el.limits.hs_bounds(K2,K1,G2,G1,c2,c1)
Vphs1 = np.sqrt((Khs1+4.0/3.0*Ghs1)/roe)
Vshs1 = np.sqrt(Ghs1/roe)
Vphs2 = np.sqrt((Khs2+4.0/3.0*Ghs2)/roe)
Vshs2 = np.sqrt(Ghs2/roe)

# Plot bounds
pmin=0
pmax=1





###############################################

# Aspect ratio of two phases
a2=1.0
# Unrelaxed, high frequency limit
a_Ku1,a_Gu1 = el.eep.dem2(K1,K2,G1,G2,a2,c2)
a_Vpu1=np.sqrt((a_Ku1+4.0/3.0*a_Gu1)/roe)
a_Vsu1=np.sqrt(a_Gu1/roe)
# Relaxed, low frequency limit
# Calculate dry moduli first (void fill)
#K,G = el.eep.mod_a(K1,zero,G1,zero,a2,c2)
# Then use Gassman's fluid substitution
#a_Kr1,a_Gr1 = el.eep.gassman_2(K,G,K1,G1,K2,G2,c2)
#a_Vpr1=np.sqrt((a_Kr1+4.0/3.0*a_Gr1)/roe)
#a_Vsr1=np.sqrt(a_Gr1/roe)

pmin=0
pmax=1
fig = plt.figure()
plt.xlabel('Melt fraction')
plt.ylabel('M (GPa)')
plt.xlim(pmin,pmax)
plt.ylim(0,90)
plt.plot(c2,Khs1,lw='1',c='r')
plt.plot(c2,Khs2,lw='1',c='r')
plt.plot(c2,Ghs1,lw='1',c='r')
plt.plot(c2,Ghs2,lw='1',c='r')
plt.plot(c2,a_Ku1,c='black',linewidth=1.5)
#plt.plot(c2,a_Kr1,c='black',linestyle='--',linewidth=1.5)
plt.plot(c2,a_Gu1,c='black',linewidth=1.5)
#plt.plot(c2,a_Gr1,c='black',linestyle='--',linewidth=1.5)
plt.show()



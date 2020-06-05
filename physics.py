# Various useful functions and relationships for rock physics
# 
# Michele Paulatto - Imperial College London - May 2018
#
# Licenced under Creative Commons Attribution 4.0 International (CC BY 4.0)
# You are free to copy, use, modify and redistribute this work provided that you 
# provide appropriate credit to the original author, provide a link to the licence 
# and indicate if changes where made.
# Full terms at: https://creativecommons.org/licenses/by/4.0/
#
# List of functions:
# Primary: vp2vs

#-------------------------------------
import numpy as np
import math

FF = 1.0	# Dependence of Q on frequency
HH = 2.76e5	# Activation enthalpy J/mol
RR = 8.314472	# Gas constant J/mol
dvpdt0 = -5.7e-1
bbvp = -8.1e-5	# K-1 anharmonic dlnVp/Dt from Dunn 2000, Christensen 1979
const = FF*HH/math.pi/RR

# Parameters for dependence of Q on temperature
alpha = 0.15
f = 10.0
d = 500.


#########################################
def ddensdt_a(dens0,t0,t1): 
# Thermal expansion coefficient
  alpha=10*1.e-6
# Calculate
  ta = t1-t0
  x = dens0*np.exp(-alpha*ta)
  return x 

#########################################
def densdm(ros,rof,fsolid,ffluid):
  x = ros*fsolid+rof*ffluid

#########################################
# Brocher's Vp to Vs relationship
def vp2vs(x):
	y = 0.7858 - 1.2344*x + 0.7949*x**2. - 0.1238*x**3. + 0.64e-2*x**4.
	return y

#########################################
# Brocher's Vp to density relationship (vp in km/s)
def vp2dens(x):
	y = 1.6612*x - 0.4721*x**2 + 0.0671*x**3 - 0.0043*x**4 + 0.000106*x**5
	return y


#########################################
def karato(Vp,T0,DT,QQ):
# Estimate Vp anomaly from T anomaly using Karato (1993)
# Some constants and variables
	nsteps=100
# Vp anomaly due to temperature change
	dT = (DT)/(nsteps*1.0)
	ttemp = T0*1.0
	dlogv=0.0
	for l in range(1,nsteps):
		ttemp = ttemp+dT
		dlnvdt = bbvp-const/QQ/ttemp**2.
		dlogv = dlogv+dlnvdt*dT
	Vpanot = Vp*np.exp(dlogv)-Vp
	return Vpanot
	
#########################################
def karato_a(Vp,T0,DT,QQ):
# Estimate Vp anomaly from T anomaly using Karato (1993)
# Analytical solution
# Vp in km/s, T in kelvin
# Vp anomaly due to temperature change
	Vp1 = Vp*np.exp(bbvp*DT)*np.exp(const/QQ*(1/(T0+DT)-1/T0))
	return Vp1

#########################################
def karato_q(Vp,T0,DT,A=1.0,Ea=8.73e5,al=0.08,qmax=2e3,C=34.7,tf=1.0):
# Estimate Vp anomaly from T anomaly using Karato (1993)
# Here Q depends on temperature as in Fontaine 2005
# Some constants and variables
	nsteps = 100
	dlnvpdt0 = dvpdt0/Vp   # vp in m/s
	B = FF*Ea/math.pi/RR
# Vp anomaly due to temperature change
	dT = (DT)/(nsteps*1.0)
	ttemp = T0*1.0
	Vp1 = Vp*1.0
	for l in range(1,nsteps):
		ttemp = ttemp+dT
		QL = qmax - (ttemp-273)*(qmax-1e2)/1050.
		QQ = A/(1./(2.25/(C*(1./(f*d)*np.exp(-Ea/(ttemp*tf)/RR))**al))+1./QL)
		Vp1 = Vp1*np.exp(dlnvpdt0*dT)*np.exp(B/QQ*(1./(ttemp+dT)-1./ttemp))
	return Vp1,QQ

#########################################
def karato_vpvs(V,T0,DT,QQ,bb,H):
# Estimate Vp anomaly from T anomaly using Karato (1993)
# Analytical solution
# Vp in km/s, T in kelvin
# Vp anomaly due to temperature change
	c = FF*H/math.pi/RR
	V1 = V*np.exp(bb*DT)*np.exp(c/QQ*(1/(T0+DT)-1/T0))
	return V1
	
#########################################
def karato_i(Vp,Vp0,T0,QQ):
# Estimate T anomaly from Vp anomaly
# Some constants and variables
	nsteps=100
	T=T0
	deltalogv = np.log(Vp)-np.log(Vp0)
	logV = np.log(Vp)
	idlogv=deltalogv/(nsteps*1.0)
	for l in range(1,nsteps):
		dtdlnv = 1./(bbvp-const/QQ/T**2.)
		T = T+idlogv*dtdlnv
	return T

#########################################
def karato_ai(Vp,Vp0,T0,QQ):
# Estimate T anomaly from Vp anomaly, analytical solution
# Some constants and variables
	t=T0    # initialize T
	dt=1    # Temperature step in search
	x = Vp0 # initialize vp
	while x > Vp:
		x = Vp0*np.exp(bbvp*(t-T0))*np.exp(const/QQ*(1./t-1./T0))
		t=t+dt
	return t


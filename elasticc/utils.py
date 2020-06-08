"""
Author: michpaulatto
GitHub: ElasticC

Various useful functions and relationships for rock physics. First written 
in 2018 by Michele Paulatto - Imperial College London
Modified June 2020
 
List of functions:
Primary: vp2vs, vs2vp, vp2dens, dens2vp, vp2t, t2vp
"""

#-------------------------------------
import numpy as np
import math
from . import constants



#########################################
# Brocher's Vp to Vs relationship vp in m/s
# Based on regression in Brocher (2005, doi:10.1785/0120050077)
def vp2vs(x):
	x = x*1e-3
	y = 0.7858 - 1.2344*x + 0.7949*x**2. - 0.1238*x**3. + 0.64e-2*x**4.
	return y*1e3

#########################################
# Brocher's Vs to Vp relationship vs in m/s
# Based on regression in Brocher (2005, doi:10.1785/0120050077)
def vs2vp(x):
	x = x*1e-3
	y = 0.9409 + 2.0947*x -0.8206*x**2 + 0.2683*x**3 - 0.0251*x**4
	return y*1e3

#########################################
# Brocher's Vp to density relationship vp in m/s, density in g/cm^3
# Based on regression in Brocher (2005, doi:10.1785/0120050077)
def vp2dens(x):
	x = x*1e-3
	y = 1.6612*x - 0.4721*x**2 + 0.0671*x**3 - 0.0043*x**4 + 0.000106*x**5
	return y*1e3

##########################################
# Inverse of Brocher's relation for density, vp in m/s, density in g/cm^3
# Based on regression in Brocher (2005, doi:10.1785/0120050077)
def dens2vp(x):
	x = x*1e-3
	y = 39.128*x - 63.064*x**2 + 37.083*x**3 - 9.1819*x**4 + 0.8228*x**5
	return y*1e3
	

#########################################
def t2vp(Vp,T0,DT,QQ):
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
def t2vp_a(Vp,T0,DT,QQ):
# Estimate Vp anomaly from T anomaly using Karato (1993)
# Analytical solution
# Vp in km/s, T in kelvin
# Vp anomaly due to temperature change
	Vp1 = Vp*np.exp(bbvp*DT)*np.exp(const/QQ*(1/(T0+DT)-1/T0))
	return Vp1



#########################################
def karato_vq(Vp,T0,DT,QQ,Ea):
# Estimate Vp anomaly from T anomaly using Karato (1993)
# Here Q is variable and provided by the user
# Some constants and variables
	nsteps = 100
	dlnvpdt0 = dvpdt0/Vp
	B = FF*Ea/math.pi/RR
# Vp anomaly due to temperature change
	dT = (DT)/(nsteps*1.0)
	ttemp = T0*1.0
	dlogv=0.0
	for l in range(1,nsteps):
		ttemp = ttemp+dT
		dlnvdt = dlnvpdt0-B/QQ/ttemp**2.
		dlogv = dlogv+dlnvdt*dT
	Vpanot = Vp*np.exp(dlogv)-Vp
	return Vpanot + Vp


def karato_qp(Vp,T0,DT,A=1.0,Ea=8.73e5,al=0.08,qmax=1e4,C=34.7,tf=1.0):
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
		QQ = A/(1./(2.25/(C*(1./(f*d)*np.exp(-Ea/(ttemp*tf)/RR))**al))+1/qmax)
		Vp1 = Vp1*np.exp(dlnvpdt0*dT)*np.exp(B/QQ*(1./(ttemp+dT)-1./ttemp))
	return Vp1,QQ

#########################################
def karato_qs(Vs,T0,DT,A=1.0,Ea=8.73e5,al=0.08,qmax=0.4444e4,C=34.7,tf=1.0):
# Estimate Vs anomaly from T anomaly using Karato (1993)
# Here Q depends on temperature as in Fontaine 2005
# Some constants and variables
	nsteps = 100
	dlnvsdt0 = dvsdt0/Vs   # vp in m/s
	B = FF*Ea/math.pi/RR
# Vp anomaly due to temperature change
	dT = (DT)/(nsteps*1.0)
	ttemp = T0*1.0
	Vs1 = Vs*1.0
	for l in range(1,nsteps):
		ttemp = ttemp+dT
		QL = qmax - (ttemp-273)*(qmax-1e2)/1050.
		QQ = A/(1./(1.0/(C*(1./(f*d)*np.exp(-Ea/(ttemp*tf)/RR))**al))+1./qmax)
		Vs1 = Vs1*np.exp(dlnvsdt0*dT)*np.exp(B/QQ*(1./(ttemp+dT)-1./ttemp))
	return Vs1,QQ

	
#########################################
def karato_qpi(Vp,Vp0,T0,A=1.0,Ea=8.73e5,al=0.08,qmax=1e4,C=34.7,):
# Estimate T anomaly from Vp anomaly
# Some constants and variables
	nsteps = 100
	B = FF*Ea/math.pi/RR
	T=T0
	deltalogv = np.log(Vp)-np.log(Vp0)
	logV = np.log(Vp)
	idlogv=deltalogv/(nsteps*1.0)
	for l in range(1,nsteps):
		QL = qmax - (T-273)*(qmax-1e2)/1050.
		QQ = A/(1./(2.25/(C*(1./(f*d)*np.exp(-Ea/T/RR))**al))+1./qmax)
		dtdlnv = 1./(bb-B/QQ/T**2.)
		T = T+idlogv*dtdlnv
	return T

#########################################
def karato_qsi(Vs,Vp0,T0,A=1.0,Ea=8.73e5,al=0.08,qmax=0.4444e4,C=34.7,):
# Estimate T anomaly from Vs anomaly
# Some constants and variables
	nsteps = 100
	B = FF*Ea/math.pi/RR
	T=T0
	deltalogv = np.log(Vs)-np.log(Vs0)
	logV = np.log(Vs)
	idlogv=deltalogv/(nsteps*1.0)
	for l in range(1,nsteps):
		QL = qmax - (T-273)*(qmax-1e2)/1050.
		QQ = A/(1./(1.0/(C*(1./(f*d)*np.exp(-Ea/T/RR))**al))+1./qmax)
		dtdlnv = 1./(bb-B/QQ/T**2.)
		T = T+idlogv*dtdlnv
	return T




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
def vp2t_i(Vp,Vp0,T0,QQ):
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
def vp2t_ai(Vp,Vp0,T0,QQ):
# Estimate T anomaly from Vp anomaly, analytical solution
# Some constants and variables
	t=T0    # initialize T
	dt=1    # Temperature step in search
	x = Vp0 # initialize vp
	while x > Vp:
		x = Vp0*np.exp(bbvp*(t-T0))*np.exp(const/QQ*(1./t-1./T0))
		t=t+dt
	return t


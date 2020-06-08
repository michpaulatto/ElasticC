"""
Author: michpaulatto
GitHub: ElasticC
Some rigorous bounds which may be handy
"""

#-------------------------------------
# The Voigt bound, i.e. the simple weighted average
def Voigt(a1,c1,a2,c2):
  x = a1*c1+a2*c2
  return x

#-------------------------------------
# The Reuss bound, i.e. the harmonic mean
def Reuss(a1,c1,a2,c2):
  if a2 > 0 and a1 > 0:
    x = 1.0/(c1/a1+c2/a2)
  else:
    x = np.zeros_like(c1) 
  return x

#-------------------------------------
# The Voigt bound for n components
def Voigt_n(x,c):
	y = np.average(x,weights=c)
	return y

#-------------------------------------
# The Reuss bound for n components
def Reuss_n(x,c):
	a = np.average(1./x,weights=c)
	y=1./a
	return y

#-------------------------------------
# Voigt-Reuss-Hill average for n components
def VRH_n(x,c):
	y1=Voigt_n(x,c)
	y2=Reuss_n(x,c)
	y=(y1+y2)/2.
	return y

#-------------------------------------
def hs_bounds(K1,K2,G1,G2,c1,c2):
# Hashin-Shtrikman bounds. We get the upper and lower bound  by swapping
# the first and second component
  Khs=K1+c2/(1./(K2-K1)+c1/(K1+4./3.*G1))
  Ghs=G1+c2/(1./(G2-G1)+(2.*c1*(K1+2.*G1))/(5.*G1*(K1+4./3.*G1)))
  return Khs, Ghs
  




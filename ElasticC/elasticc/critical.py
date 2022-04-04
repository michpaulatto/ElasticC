# Elastic moduli calculation using critical porosity
# Michele Paulatto - Imperial College London - January 2018
# Based on Chu et al., 2010 and Nur et al., 1998
#
# Licenced under Creative Commons Attribution 4.0 International (CC BY 4.0)
# You are free to copy, use, modify and redistribute this work provided that you 
# provide appropriate credit to the original author, provide a link to the licence 
# and indicate if changes where made.
# Full terms at: https://creativecommons.org/licenses/by/4.0/



import numpy as np

def mod_c(Km,Kl,Gm,Gl,phic,ca):
  nm=np.shape(ca)[0]
# Initialize arrays as simple average 
  Kd=np.linspace(Km,Kl,num=nm)
  Gd=np.linspace(Gm,Gl,num=nm)
  for j in range(nm):
    c=ca[j]
    if c >= phic:
      Kd[j] = 0.0
      Gd[j] = 0.0
    else:
      Ki=0.0
      Kd[j] = (Km-Ki)*(1.0-c/phic)+Ki
      Gd[j] = Gm*(1.0-c/phic)
  Ke,Ge = gassman_2(Kd,Gd,Km,Gm,Kl,Gl,ca)
  return Ke, Ge
  
#-------------------------------------
def gassman_2(Kd,Gd,Km,Gm,Kl,Gl,p):
# I follow here the equation published by Marion 1990 PhD Thesis
# Km, Gm = moduli of solid matrix
# Kl, Gl = moduli of fluid
# Kd, Gd = moduli of dry composite
# p = porosity
# Apply Gassman's equation
#  Kw = Kd
  a = Kd/(Km-Kd)+Kl/(Km-Kl)/p
  Kw = Km*a/(1+a)
  Gw = Gd
  return Kw, Gw






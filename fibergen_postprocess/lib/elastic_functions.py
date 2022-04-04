# ************************************************************************* 
'''Python functions for manipulating stiffnes matrix'''
# ************************************************************************* 
#
# Original code from: 
#    Kim, E., Kim, Y., and Mainprice, D. (2019), "GassDem: A MATLAB program
#    for modeling the anisotropic seismic properties of porous medium 
#    using differential effective medium theory and Gassmann's poroelastic 
#    relationship", Computers and Geosciences.
# Github repo:
#   https://github.com/ekim1419/GassDem

import numpy as np
from math import pi, cos, sin, tan, acos, atan, asin, sqrt
import sys


#**************************************************************************

def Kelvin2Voigt(Kelvin):
    """ 
    Converts Kelvin to Voigt
    
    Parameters
    --------------
    Kelvin : numpy array
        Kelvin matrix
    
    Returns
    -------------
    Voigt : numpy array
        Voigt Matrix
    """
   
    Kelvin2VoigtMatrix = np.array([[1, 0, 0, 0, 0, 0], \
                     [0, 1, 0, 0, 0, 0], \
                     [0, 0, 1, 0, 0, 0], \
                     [0, 0, 0, 1/sqrt(2), 0, 0], \
                     [0, 0, 0, 0, 1/sqrt(2), 0], \
                     [0, 0, 0, 0, 0, 1/sqrt(2)]])
    
    # Matrix Kelvin to Voigt method  
    Voigt = Kelvin2VoigtMatrix*Kelvin*Kelvin2VoigtMatrix
    return Voigt

#**************************************************************************

def Voigt2Kelvin(Voigt):
    """ 
    Converts Voigt to Kelvin
    
    Parameters
    --------------
    Voigt : numpy array
        Voigt Matrix
    
    Returns
    -------------
    Kelvin : numpy array
        Kelvin matrix
    """

    Voigt2KelvinMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], \
                     [0, 0, 1, 0, 0, 0], [0, 0, 0, sqrt(2), 0, 0], \
                     [0, 0, 0, 0, sqrt(2), 0], [0, 0, 0, 0, 0, sqrt(2)]])
    # Matrix Voigt to Kelvin method  
    Kelvin = Voigt2KelvinMatrix*Voigt*Voigt2KelvinMatrix
    return Kelvin

#**************************************************************************

def gasman(Cdry,Cmin,Bf,pore,Csat):
    """
    Gassmann's (1951) poroelastic relation extended to anisotropic media
    by R.Brown and J.Korringa (1975) Geophysics vol.40, pp.608-616. 
    
    Parameters
    -----------------
    Cdry : numpy array
        Cij of dry porous media
    Cmin : numpy array
        Cij of non-porous media
    Bf : numpy array
        Beta of pore fluid (1/Kf)
    pore : float
        Porosity (range 0 to 1.0)
    Csat : numpy array
        Cij of saturated media
    
    Returns
    -----------------
    Csat : numpy array
        Cij of saturated media
    """

    Sdry   = np.zeros([6,6])
    Smin   = np.zeros([6,6])
    Ssat   = np.zeros([6,6])
    Sdry4  = np.zeros([3,3,3,3])
    Smin4  = np.zeros([3,3,3,3])
    Ssat4  = np.zeros([3,3,3,3]) 
    Sdiff4 = np.zeros([3,3,3,3])

    Sdry[:,:] = Cdry[:,:]
    Smin[:,:] = Cmin[:,:]

    # invert Cij to Sij
    Sdry = inv(Sdry) # GPa 
    Smin = inv(Smin) # GPa

    # convert compilance to 4 index
    # Sij -> Sijkl
    Sdry4, _, _, _ = stiffness(Sdry4,Sdry,3)
    Smin4, _, _, _ = stiffness(Smin4,Smin,3)

    # beta = 1/K for dry composite and non-porous
    Bdry = 0
    Bmin = 0
    
    for alpha in range(0,3):
        for beta in range(0,3):
            Bdry = Bdry + Sdry4[alpha,alpha,beta,beta]
            Bmin = Bmin + Smin4[alpha,alpha,beta,beta]
    
    
    # dividing constant
    s = (Bdry-Bmin)+(Bf-Bmin)*pore
    
    # Sijkl loop
    for i in range(0,3):
        for j in range(0,3):
            # sum over i,j,alpha,alpha
            Sijdry = 0
            Sijmin = 0
            for alpha in range(0,3):
                Sijdry = Sijdry + Sdry4[i,j,alpha,alpha]
                Sijmin = Sijmin + Smin4[i,j,alpha,alpha]
            
            for k in range(0,3): 
                for l in range(0,3):
                #  sum over k,l,alpha,alpha
                    Skldry = 0
                    Sklmin = 0
                    for alpha in range(0,3):
                        Skldry = Skldry + Sdry4[k,l,alpha,alpha]
                        Sklmin = Sklmin + Smin4[k,l,alpha,alpha]
                    
                    # difference Sdryijkl - Satijkl
                    Sdiff4[i,j,k,l] = ((Sijdry-Sijmin)*(Skldry-Sklmin))/s


    # Ssat = Sdry - Sdiff
    Ssat4[:,:,:,:] = Sdry4[:,:,:,:] - Sdiff4[:,:,:,:]


    # convert compilance to 2 index
    # Sijkl -> Sij (Mode = 1)
    _,Ssat,_,_ = stiffness(Ssat4,Ssat,1)

    # invert Sij to Cij
    Csat[:,:] = inv(Ssat)

    return Csat

#**************************************************************************

def stiffness(a4,b2,mode):
    """ Converts stiffnesses (Cij) and compliances (Sij) between notations
    
    Parameters
    ---------------
    a4 : numpy array 
        Matrix in 4-Subscript notation
    b2 : numpy array
        Matrix in 2-Subscript notation
    mode : int
        = 1  Compliance  Sijkl -> Sij
        = 2  Stiffness   Cijkl -> Cij
        = 3  Compliance    Sij -> Sijkl
        = 4  Stiffness     Cij -> Cijkl
    
    Returns
    --------------
    a4, b2, mode
    ierr : int
        = 16 UNDEFINED MODE
    
    """
    # ref. K.Helbig 1994 Foundations of Anisotropy for Exploration Seismics
    # Pergamon pp.123-124.

    ierr = 0

    # Block off with non-existing code
    if mode < 1 or mode > 4:
        ierr = 16


    # Set output to zero
    if mode <= 2:
        b2 = np.zeros([6,6])


    if mode >= 3:
        a4 = np.zeros([3,3,3,3])


    # Set subscript exchange and factors
    for i in range(0,3): 
        for j in range(0,3):
            if i == j:
                p = i
                f1 = 1
            else:
                p = 6-i-j
                f1 = 2

            for k in range(0,3):
                for l in range(0,3): 
                    if k == l:
                        q = k
                        f2 = 1
                    else:
                        q = 6-k-l
                        f2 = 2
                
                    # Conversion 4 -> 2
                    if mode == 1:
                        b2[p,q] = a4[i,j,k,l]*f1*f2

                    if mode == 2:
                        b2[p,q] = a4[i,j,k,l]

                    # Conversion 2 -> 4    
                    if mode == 3:
                        a4[i,j,k,l] = b2[p,q]/f1/f2

                    if mode == 4:
                        a4[i,j,k,l] = b2[p,q]
    return a4,b2,mode,ierr

#**************************************************************************

def xyzv(c, drock, ijkl):
    """ Phase velocities in X,Y,Z directions
    
    Parameters
    -----------------
    c :  numpy array
        Elastic stiffness tensor
    drock : float
        Density of rock
    ijkl : numpy array 
    Returns
    -----------------
    bulk : float
        Isotropic Bulk modulus (GPa)
    shear : float
        Isotropic Shear modulus (GPa)
    VpX : float
        P - wave velocity X direction
    Vs1X : float 
        S - wave velocity in X direction (fast)
    Vs2X : float
        S - wave velocity in X direction (slow)
    VpY : float 
        P - wave velocity Y direction
    Vs1Y : float 
        S - wave velocity in Y direction (fast)
    Vs2Y : float 
        S - wave velocity in Y direction (slow)
    VpZ : float
        P - wave velocity Z direction 
    Vs1Z : float
        S - wave velocity in Z direction (fast) 
    Vs2Z : float 
        S - wave velocity in Z direction (slow)
    """

    # X = lineation (East) = 010
    # Y = normal to lineation (Origin) = 001
    # Z = foliation pole (North) = 100

    xi = np.zeros([1,3])

    xi[0,0] = 1
    xi[0,1] = 0
    xi[0,2] = 0

    V,_ = velo2(xi,drock,c,ijkl)

    VpZ  = V[0,0]
    Vs1Z = V[0,1]
    Vs2Z = V[0,2]

    xi[0,0] = 0
    xi[0,1] = 1
    xi[0,2] = 0

    V,_ = velo2(xi,drock,c,ijkl)
    
    VpX  = V[0,0]
    Vs1X = V[0,1]
    Vs2X = V[0,2]

    xi[0,0] = 0
    xi[0,1] = 0
    xi[0,2] = 1

    V,_ = velo2(xi,drock,c,ijkl)

    VpY  = V[0,0]
    Vs1Y = V[0,1]
    Vs2Y = V[0,2]

    # Directionally averaged velocities
    Vp = (VpX + VpY + VpZ)/3
    Vs = (Vs1X + Vs2X + Vs1Y + Vs2Y + Vs1Z + Vs2Z)/6

    # Directionally averaged elastic constants (GPa)
    c11 = drock*(Vp**2)
    c44 = drock*(Vs**2)

    # Isotropic shear and bulk moduli (GPa)
    shear = c44
    bulk  = c11 - 4*c44/3

    return bulk,shear,VpX,Vs1X,Vs2X,VpY,Vs1Y,Vs2Y,VpZ,Vs1Z,Vs2Z

#**************************************************************************

def velo2(x,rho,c,ijkl):
    
    """ Phase-velocity surfaces in an anisotropic medium
    
    Parameters
    ----------------
    x(3) : numpy array
        Direction of interest
    rho : float
        Density [g/cm3]
    c : numpy array
        Elastic stiffness tensor [GPa]
    ijkl : nunpy array
    Returns 
    -----------------
    v : numpy array
        Phase velocities (1,2,3 = P,S,SS) [km/s]
    eigvec : numpy array
        Eigenvectors stored by columns """

    c = np.transpose(c)
    v      = np.zeros([1,3])
    t      = np.zeros([3,3])
    eigvec = np.zeros([3,3])

    # Form symmetric matrix Tik = Cijkl*Xj*Xl
    for i in range(0,3): 
        for k in range(0,3):
            t[i,k] = 0
            for j in range(0,3):
                for l in range(0,3):
                    m = ijkl[i,j]
                    n = ijkl[k,l]
                    t[i,k] = t[i,k] + c[m,n]*x[0,j]*x[0,l]



    # Determine the eigenvalues of symmetric Tij
    ei,evalj = jacobi(t,3,3)

    for i in range(0,3): 
        for j in range(0,3):
            eigvec[i,j] = ei[j,i]


    # X(I) = wave normal
    # Eigenvalues  = function of wave velocity
    # Eigenvectors = displacement vector
    # polarization plane = contains wave normal and displacement vector
    for i in range(0,3): 
        de = evalj[0,i]      # [GPa]
        dv = sqrt(de/rho) # [km/s]
        v[0,i] = dv

    # sort velocities into ascending order
    for iop in range(0,2): 
        nop = iop + 1
        for inop in range(nop,3): 
            if v[0,inop] <= v[0,iop]:
                continue

            value   = v[0,iop]
            v[0,iop]  = v[0,inop]
            v[0,inop] = value
            for m in range(0,3): 
                val = eigvec[iop,m]
                eigvec[iop,m] = eigvec[inop,m]
                eigvec[inop,m] = val


    eigvec = np.transpose(eigvec)

    return v,eigvec

#**************************************************************************

def jacobi(A,N,NP):
    """ Computes all eigenvalues and vectors of a real symmetric matrix A(i,j)
    
    Parameters
    ------------------
    A : numpy array
        Sorted in physical array A(NP,NP) on output the elements of A above the diagonal are destroyed
    N : int
    NP : int
    
    Returns
    -----------------
    D : numpy array 
        Eigenvalues of A in its first N elements
    V : numpy array
        Columns contain normalized vectors
    """

    A = np.transpose(A)
    nmax = 100
    D = np.zeros([1,NP])
    V = np.zeros([NP,NP])
    b = np.zeros([1,nmax])
    z = np.zeros([1,nmax])


    for ip in range(0,N):
        for iq in range(0,N): 
            V[ip,iq] = 0

        V[ip,ip] = 1

    for ip in range(0,N): 
        b[0,ip] = A[ip,ip]
        D[0,ip] = b[0,ip]
        z[0,ip] = 0


    nrot = 0

    for i in range(1,51):
        sm = 0
        for ip in range(0,N-1): 
            for iq in range(ip+1,N):
                sm = sm + abs(A[ip,iq]) 

        if sm == 0:
            return V,D 

        if i < 4:
            tresh = 0.2*sm/(N**2)
        else:
            tresh = 0

        for ip in range(0,N-1): 
            for iq in range(ip+1,N):    
                G = 100*abs(A[ip,iq])
                if i > 4 & (abs(D[0,ip])+G == abs(D[0,ip])) & (abs(D[0,iq])+G == abs(D[0,iq])):
                    A[ip,iq] = 0
                elif abs(A[ip,iq]) > tresh:
                    h = D[0,iq] - D[0,ip]
                    if abs(h)+G == abs(h):
                        t = A[ip,iq]/h
                    else:
                        theta = 0.5*h/A[ip,iq]
                        t = 1/(abs(theta) + sqrt(1+(theta**2)))
                        if theta < 0:
                            t = -t

                    c = 1/sqrt(1+t**2)
                    s = t*c
                    tau = s/(1+c)
                    h = t*A[ip,iq]
                    z[0,ip] = z[0,ip] - h
                    z[0,iq] = z[0,iq] + h
                    D[0,ip] = D[0,ip] - h
                    D[0,iq] = D[0,iq] + h
                    A[ip,iq] = 0
                    for j in range(0,ip-1): 
                        G = A[j,ip]
                        h = A[j,iq]
                        A[j,ip] = G - s*(h+G*tau)
                        A[j,iq] = h + s*(G-h*tau)

                    for j in range(ip+1,iq): 
                        G = A[ip,j]
                        h = A[j,iq]
                        A[ip,j] = G - s*(h+G*tau)
                        A[j,iq] = h + s*(G-h*tau)

                    for j in range(iq+1, N): 
                        G = A[ip,j]
                        h = A[iq,j]
                        A[ip,j] = G - s*(h+G*tau)
                        A[iq,j] = h + s*(G-h*tau)

                    for j in range(0,N): 
                        G = V[j,ip]
                        h = V[j,iq]
                        V[j,ip] = G - s*(h+G*tau)
                        V[j,iq] = h + s*(G-h*tau)

                    nrot = nrot + 1


        for ip in range(0,N): 
            b[0,ip] = b[0,ip] + z[0,ip]
            D[0,ip] = b[0,ip]
            z[0,ip] = 0


    V = np.transpose(V)

    print('50 iterations should never happen')
    return V,D

#**************************************************************************

def xyzv3(c,drock,ijkl):
    """ Phase velocities in X,Y,Z directions
    
    Parameters
    -----------------
    c :  numpy array
        Elastic stiffness tensor
    drock : float
        Density of rock
    ijkl : numpy array 
    Returns
    -----------------
    bulk : float
        Isotropic Bulk modulus (GPa)
    shear : float
        Isotropic Shear modulus (GPa)
    
    CVpX,CVs1X,CVs2X,CVpY,CVs1Y,CVs2Y,CVpZ,CVs1Z,CVs2Z : float
        Seismic elastic moduli (GPa) in directions X,Y,Z
    """

    # X = lineation (East) = 010
    # Y = normal to lineation (Origin) = 001
    # Z = foliation pole (North) = 100

    xi = np.zeros([1,3])

    xi[0,0] = 1
    xi[0,1] = 0
    xi[0,2] = 0

    V,_ = velo2(xi,drock,c,ijkl)

    VpZ  = V[0,0]
    Vs1Z = V[0,1]
    Vs2Z = V[0,2]

    xi[0,0] = 0
    xi[0,1] = 1
    xi[0,2] = 0

    V,_ = velo2(xi,drock,c,ijkl)

    VpX  = V[0,0]
    Vs1X = V[0,1]
    Vs2X = V[0,2]

    xi[0,0] = 0
    xi[0,1] = 0
    xi[0,2] = 1

    V,_ = velo2(xi,drock,c,ijkl)

    VpY  = V[0,0]
    Vs1Y = V[0,1]
    Vs2Y = V[0,2]

    # Directionally averaged velocities
    Vp = (VpX + VpY + VpZ)/3
    Vs = (Vs1X + Vs2X + Vs1Y + Vs2Y + Vs1Z + Vs2Z)/6

    # Directionally averaged elastic constants (GPa)
    c11 = drock*Vp**2
    c44 = drock*Vs**2

    # Isotropic shear and bulk moduli (GPa)
    shear = c44
    bulk  = c11 - 4*c44/3

    # Seismic elastic moduli (GPa) in directions X,Y,Z
    CVpX  = drock*VpX**2
    CVs1X = drock*Vs1X**2
    CVs2X = drock*Vs2X**2
    CVpY  = drock*VpY**2
    CVs1Y = drock*Vs1Y**2
    CVs2Y = drock*Vs2Y**2
    CVpZ  = drock*VpZ**2
    CVs1Z = drock*Vs1Z**2
    CVs2Z = drock*Vs2Z**2

    return bulk,shear,CVpX,CVs1X,CVs2X,CVpY,CVs1Y,CVs2Y,CVpZ,CVs1Z,CVs2Z

#**************************************************************************

def fmax3(CVpX,CVpY,CVpZ,visc,Vpf,rhof):
    """ Calculates fmax from Vp in X,Y,Z direction
    Parameters
    ---------------------
    CVpX,CVpY,CVpZ : float
        Seismic elastic moduli (GPa) in directions X,Y,Z
    visc : float
        Viscosity of fluid (Pa.s)
    Vpf : float
        Vp of fluid (km/s)
    rhof : float
        Fluid density (g/cm3)
    Returns 
    ---------------------
    fVpX,fVpY,fVpZ : float
        fmax (1/s) associated with VpX,VpY,VpZ  """

    # Seismic Vp -> elastic moduli of fluid in GPa
    CVpf = rhof*Vpf**2

    # Seismic moduli in GPa -> Pa
    Xmodd = CVpX*10**9
    Ymodd = CVpY*10**9
    Zmodd = CVpZ*10**9
    Fmodd = CVpf*10**9

    # fmax
    fVpX = sqrt(Fmodd*(Fmodd+Xmodd))/visc
    fVpY = sqrt(Fmodd*(Fmodd+Ymodd))/visc
    fVpZ = sqrt(Fmodd*(Fmodd+Zmodd))/visc

    return fVpX,fVpY,fVpZ

#**************************************************************************

def qinv(rho,VpL,Vs1L,Vs2L,VpH,Vs1H,Vs2H):
    """ Calculates the inverse of Qmax
    Parameters
    -----------------
    rho : float
        Density (g/cm3)
    VpL,Vs1L,Vs2L : float
        Low frequency velocities [km/s]
    VpH,Vs1H,Vs2H : float
        High frequency velocities [km/s]
    Returns
    --------------------
    Qp,Qs1,Qs2 : float
        1/Qmax
    """

    # Elastic constants (GPa) associated with Vp,Vs1 and Vs2
    c11l  = rho*VpL**2
    c44l1 = rho*Vs1L**2
    c44l2 = rho*Vs2L**2
    c11h  = rho*VpH**2
    c44h1 = rho*Vs1H**2
    c44h2 = rho*Vs2H**2

    # 1/Qp max
    Qp = (c11h-c11l)/(2*sqrt(c11h*c11l))

    # 1/Qs max
    Qs1 = (c44h1-c44l1)/(2*sqrt(c44h1*c44l1))
    Qs2 = (c44h2-c44l2)/(2*sqrt(c44h2*c44l2))

    return Qp,Qs1,Qs2

#**************************************************************************

def vpfile(c,drock,ihemi,icont,outputDir,title,textvf,ijkl):
    
    """ Writes VpG files
    
    Writes:
    Phase velocities qVp,qVS1,qVS2
    Particle motion vectors for qVp,qVS1,qVS2
    Group velocity vectors for qVp,qVS1,qVS2
    """

    sn1 = np.zeros([1,3])
    sn2 = np.zeros([1,3])
    sn3 = np.zeros([1,3])
    pm1 = np.zeros([1,3])
    pm2 = np.zeros([1,3])
    pm3 = np.zeros([1,3])
    c4  = np.zeros([3,3,3,3])

    #     Convert from Cij to Cijkl
    c4,_,_,_ = stiffness(c4,c,4)

    #     PLANE WAVE CALCULATION OVER WHOLE GEOGRAPHIC HEMISPHERE    
    #     USING A GRID SPACING OF 6 DEGREES                          
    # FILE=Vfile
    
    fid_30 = open('Clow%s_VpG.txt' %(textvf),'w+')
    fid_30.write('_Clow%s_VpG.txt \n' %(textvf))
    
    for i in range(1,17): 
        xinc = icont*(90-((i-1)*6))
        for j in range(1, 62): 
            az = 90*(j-1)/15

            # Convert geographic coordinates to direction cosines
            xi = cart(xinc,az,ihemi)

            # Calculate phase velocities 
            v,eigvec = velo2(xi,drock,c,ijkl)

            # Particle motion vectors for qP,qVS1,qVS2
            for k in range(0,3): 
                pm1[0,k] = eigvec[0,k]
                pm2[0,k] = eigvec[1,k]
                pm3[0,k] = eigvec[2,k]


            # Slowness vectors SN1,2,3 in s/Km
            snm1 = 1/v[0,0]
            snm2 = 1/v[0,1]
            snm3 = 1/v[0,2]
            for k in range(0,3): 
                sn1[0,k] = snm1*xi[0,k]
                sn2[0,k] = snm2*xi[0,k]
                sn3[0,k] = snm3*xi[0,k]


            # Ray-velocity vector corresponding to the slowness vectors SN1,2,3
            vg1 = rayvel(c4,sn1,drock)
            vg2 = rayvel(c4,sn2,drock)
            vg3 = rayvel(c4,sn3,drock)

            # Magnitude of group velocity
            vg1m = sqrt(vg1[0,0]**2 + vg1[0,1]**2 + vg1[0,2]**2)
            vg2m = sqrt(vg2[0,0]**2 + vg2[0,1]**2 + vg2[0,2]**2)
            vg3m = sqrt(vg3[0,0]**2 + vg3[0,1]**2 + vg3[0,2]**2)

            # WRITE TO FILE 15
            # Phase velocities qVp,qVS1,qVS2
            fid_30.write('%12.4f \t %12.4f \t %12.4f \t' %(v[0,0],v[0,1],v[0,2]))

            # Particle motion vectors for qVp,qVS1,qVS2
            # XP,XS1,XS2,YP,YS1,YS2,ZP,ZS1,ZS2
            fid_30.write('%12.4f \t %12.4f \t %12.4f \n %12.4f \t %12.4f \t %12.4f %12.4f \t %12.4f \t %12.4f \n' %(eigvec[0,0],eigvec[0,1],eigvec[0,2],eigvec[1,0],eigvec[1,1],eigvec[1,2],eigvec[2,0],eigvec[2,1],eigvec[2,2]))

            # Group velocity vectors for qVp,qVS1,qVS2
            fid_30.write("%12.4f \t %12.4f \t %12.4f \t %12.4f \t %12.4f \t %12.4f \t %12.4f \t %12.4f \t %12.4f \n" %(vg1[0,0],vg1[0,1],vg1[0,2],vg2[0,0],vg2[0,1],vg2[0,2],vg3[0,0],vg3[0,1],vg3[0,2]))
            
    # CLOSE OUTPUT FILE 30 Vp file
    fid_30.close()

    return v,eigvec,vg1,vg2,vg3

#**************************************************************************

def cart(xinc,azm,ihemi):
    """ Convert from spherical to cartesian coordinates
    NORTH X=100  WEST Y=010 UP Z=001 DOWN -Z=00-1
    IHEMI=-1 LOWER HEMISPHERE +VE DIP
    IHEMI=+1 UPPER HEMISPHERE +VE DIP """

    x = np.zeros([1,3])

    rad  = pi/180
    caz  = cos(azm*rad)
    saz  = sin(azm*rad)
    rinc = ihemi*xinc
    cinc = cos(rinc*rad)
    sinc = sin(rinc*rad)
    x[0,0] = caz*cinc
    x[0,1] = -saz*cinc
    x[0,2] = sinc

    # Normalize to direction cosines
    r = sqrt(x[0,0]*x[0,0] + x[0,1]*x[0,1] + x[0,2]*x[0,2])

    x[:,:] = x[:,:]/r

    return x

#**************************************************************************

def rayvel(c4,sn,rho):
    """Calculate the ray-velocity vector corresponding to a slowness vector
    
    Parameters
    ----------------
    c4 : numpy array
        Stiffness tensor
    sn : numpy array
        Slowness vector
    rho : float
        Density
    
    Returns
    -----------------
    vg : numpy array
        Ray velocity vector
    """

    # f(i,k)    - Components of determinant in terms of components of the slowness vector
    # df(i,j,k) - Array of derivatives of f with respect to the components of the slowness vector
    # dfd(i)    - Gradient of the function f

    f   = np.zeros([3,3])
    cf  = np.zeros([3,3])
    df  = np.zeros([3,3,3])
    dfd = np.zeros([1,3])
    vg  = np.zeros([1,3])

    for i in range(0,3): 
        for k in range(0,3): 
            f[i,k] = 0
            for j in range(0,3):
                for l in range(0,3):
                    # EQN. 4.23
                    f[i,k] = f[i,k] + c4[i,j,k,l]*sn[0,j]*sn[0,l]

            if i == k:
                f[i,k] = f[i,k] - rho


    # Signed cofactors of f(i,k)
    cf[0,0] = f[1,1]*f[2,2] - f[1,2]**2
    cf[1,1]= f[0,0]*f[2,2] - f[0,2]**2
    cf[2,2] = f[0,0]*f[1,1] - f[0,1]**2
    cf[0,1] = f[1,2]*f[2,0] - f[1,0]*f[2,2]
    cf[1,0] = cf[0,1]
    cf[1,2] = f[2,0]*f[0,1] - f[2,1]*f[0,0]
    cf[2,1] = cf[1,2]
    cf[2,0] = f[0,1]*f[1,2] - f[0,2]*f[1,1]
    cf[0,2] = cf[2,0]

    # Derivatives of determinant elements 
    for i in range(0,3): 
        for j in range(0,3): 
            for k in range(0,3):
                df[i,j,k] = 0
                for l in range(0,3):
                    df[i,j,k] = df[i,j,k] + (c4[i,j,k,l] + c4[k,j,i,l])*sn[0,l]


    # Components of gradient
    for k in range(0,3): 
        dfd[0,k] = 0
        for i in range(0,3): 
            for j in range(0,3):
                dfd[0,k] = dfd[0,k] + df[i,j,k]*cf[i,j]


    # Normalize to obtain group velocity vg(i)
    denom = 0
    for i in range(0,3): 
        denom = denom + sn[0,i]*dfd[0,i]


    vg[:,:] = dfd[:,:]/denom

    return vg



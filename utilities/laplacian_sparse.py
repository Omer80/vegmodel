# -*- coding: utf-8 -*-
"""
#  laplacian_sparse.py
#  
#  Copyright 2016 Omer Tzuk <omertz@post.bgu.ac.il>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

"""  
__version__=1.0
__author__ = """Omer Tzuk (omertz@post.bgu.ac.il)"""

def create_laplacian_skelton(n,bc = "neumann"):
    """ Construct laplacian sparse matrix :
    n = tuple(nx,ny), bc = "neumann"/"periodic" 
    """
#    import deepdish as dd
#    saved_laplacians=dd.io.load("laplacians.hdf5")
#    sizes=saved_laplacian['sizes']
    import numpy as np
    from scipy.sparse import diags
#    nvar = len(coeff.shape)
    if len(n) == 1:
        print "Creating 1D laplacian"
        nx = n[0]
        main_diag = np.ones(nx)*-2.0
        side_diag = side_diag = np.ones(nx-1)
        if bc == "periodic":
            loop = [1.0]
            diagonals = [main_diag,side_diag,side_diag,loop,loop]
            offsets = [0, -1, 1,-(nx-1),(nx-1)]
        elif bc == "neumann":
            up_side = side_diag.copy()
            low_side = side_diag.copy()
            up_side[0] = 2.0 
            low_side[-1] = 2.0
            diagonals = [main_diag,low_side,up_side]
            offsets = [0, -1, 1]
    elif len(n) == 2:
        print "Creating 2D laplacian"
        nx = n[0]
        ny = n[1]
        N  = nx*ny
        main_diag = np.ones(N)*-4.0
        side_diag = np.ones(N-1)
        side_diag[np.arange(1,N)%nx==0] = 0
        up_down_diag = np.ones(N-nx)
        if bc == "periodic":
            side_loops = np.zeros(N-(nx-1))
            side_loops[np.arange(0,N-(nx-1))%nx==0] = 1
            up_loops = np.ones(nx)
            diagonals = [main_diag,side_diag,side_diag,side_loops,side_loops,up_down_diag,up_down_diag,up_loops,up_loops]
            offsets = [0, -1, 1,(nx-1),-(nx-1),(nx),-(nx),(N-nx),-(N-nx)]
        elif bc == "neumann":
            up_side = side_diag.copy()
            dn_side = side_diag.copy()
            up_neu = up_down_diag.copy()
            up_side[np.arange(0,N-1)%nx==0] = 2
            dn_side[np.arange(2,N+1)%nx==0] = 2
            up_neu[np.arange(0,(N-nx))//nx==0] = 2
            dn_neu = up_neu[::-1]
            diagonals = [main_diag,dn_side,up_side,up_neu,dn_neu]
            offsets = [0, -1, 1,(nx),-(nx)]
    return diags(diagonals, offsets, format="csr")

def create_Crank_Nicolson_matrices(n,l,dt,bc = "neumann",coeff = [1.0]):
    """ Construct laplacian sparse matrix :
    n = tuple(nx,ny), bc = "neumann"/"periodic" 
    """
    import numpy as np
    from scipy.sparse import diags, block_diag
    dg  = tuple([ll/float(nn) for ll,nn in zip(l,n)])
    dx  = dg[0]
    dx2 = dx**2
    coeff=np.array(coeff)
    r_list  = coeff*dt/(2.0*dx2)
#    nvar = len(coeff.shape)
    LM = []
    RM = []
    for r in r_list:
        print "Creating 1D Crank Nicolson matrices"
        nx = n[0]
        LM_main_diag = (1.0+2.0*r)*np.ones(nx) 
        LM_side_diag = (-r)*np.ones(nx-1)
        LMdiags = [LM_main_diag,LM_side_diag,LM_side_diag]
        RM_main_diag = (1.0-2.0*r)*np.ones(nx)
        RM_side_diag = (r)*np.ones(nx-1)
        RMdiags = [RM_main_diag,RM_side_diag,RM_side_diag]
        offsets = [0, -1, 1]
        LM.append(diags(LMdiags, offsets, format="csr"))
        RM.append(diags(RMdiags, offsets, format="csr"))
    LM = block_diag(LM,format="csr")
    RM = block_diag(RM,format="csr")
    return LM,RM

def transform_laplacian_to_Crank_Nicolson(laplacian,dt):
    """
    """
    from scipy.sparse import identity,csr_matrix,csc_matrix
    I=identity(laplacian.shape[0])
    LM = csc_matrix(I - 0.5*dt*laplacian)
    RM = csr_matrix(I + 0.5*dt*laplacian)
    return LM,RM

def create_laplacian(n,l, bc = "neumann" , coeff = [1.0],verbose=False):
    """ Construct laplacian sparse matrix :
    n = tuple(nx,ny), bc = "neumann"/"periodic" 
    """
    import numpy as np
    from scipy.sparse import diags, block_diag,bmat,csr_matrix
    dx2 = (float(l[0])/float(n[0]))**2
    coeff=np.array(coeff)
#    nvar = len(coeff.shape)
    if len(n) == 1:
        if verbose:
            print "Creating 1D laplacian"
        nx = n[0]
        main_diag = np.ones(nx)*-2.0
        side_diag = side_diag = np.ones(nx-1)
        if bc == "periodic":
            loop = [1.0]
            diagonals = [main_diag,side_diag,side_diag,loop,loop]
            offsets = [0, -1, 1,-(nx-1),(nx-1)]
        elif bc == "neumann":
            up_side = side_diag.copy()
            low_side = side_diag.copy()
            up_side[0] = 2.0 
            low_side[-1] = 2.0
            diagonals = [main_diag,low_side,up_side]
            offsets = [0, -1, 1]
    elif len(n) == 2:
        if verbose:
            print "Creating 2D laplacian"
        nx = n[0]
        ny = n[1]
        N  = nx*ny
        main_diag = np.ones(N)*-4.0
        side_diag = np.ones(N-1)
        side_diag[np.arange(1,N)%nx==0] = 0
        up_down_diag = np.ones(N-nx)
        if bc == "periodic":
            side_loops = np.zeros(N-(nx-1))
            side_loops[np.arange(0,N-(nx-1))%nx==0] = 1
            up_loops = np.ones(nx)
            diagonals = [main_diag,side_diag,side_diag,side_loops,side_loops,up_down_diag,up_down_diag,up_loops,up_loops]
            offsets = [0, -1, 1,(nx-1),-(nx-1),(nx),-(nx),(N-nx),-(N-nx)]
        elif bc == "neumann":
            up_side = side_diag.copy()
            dn_side = side_diag.copy()
            up_neu = up_down_diag.copy()
            up_side[np.arange(0,N-1)%nx==0] = 2
            dn_side[np.arange(2,N+1)%nx==0] = 2
            up_neu[np.arange(0,(N-nx))//nx==0] = 2
            dn_neu = up_neu[::-1]
            diagonals = [main_diag,dn_side,up_side,up_neu,dn_neu]
            offsets = [0, -1, 1,(nx),-(nx)]
    laplacian = diags(diagonals, offsets, format="csr")
    if len(coeff.shape)==1:
        blocks = [laplacian*coeff[i] for i in xrange(coeff.shape[0])]
        laplacian = block_diag(blocks)
    elif len(coeff.shape)==2:
        blocks=[]
        for i in range(coeff.shape[0]):
            row=[]
            for j in range(coeff.shape[1]):
                row.append(laplacian*coeff[i,j])
            blocks.append(row)
        laplacian=bmat(blocks, format="csr")
    else:
        laplacian=laplacian*coeff[0]
    return csr_matrix(laplacian/dx2)

def create_gradient(n,l, bc = "neumann" , coeff = [1.0]):
    """ Construct laplacian sparse matrix :
    n = tuple(nx,ny), bc = "neumann"/"periodic" 
    """
    import numpy as np
    from scipy.sparse import diags, block_diag,bmat,csr_matrix
    dx = float(l[0])/float(n[0])
    coeff=np.array(coeff)
#    nvar = len(coeff.shape)
    if len(n) == 1:
        print "Creating 1D gradient"
        nx = n[0]
        main_diag = np.ones(nx)*(-1.0)
        side_diag = side_diag = np.ones(nx-1)
        if bc == "periodic":
            loop = [1.0]
            diagonals = [main_diag,side_diag,loop]
            offsets = [0, 1,-(nx-1)]
        elif bc == "neumann":
            up_side = side_diag.copy()
            low_side = side_diag.copy()
            up_side[0] = 0.0 
            low_side[-1] = 0.0
            main_diag[0]=0
            main_diag[-1]=0
            diagonals = [main_diag,up_side]
            offsets = [0, 1]
    elif len(n) == 2:
        print "Creating 2D gradient"
        nx = n[0]
        ny = n[1]
        N  = nx*ny
        main_diag = np.ones(N)*-4.0
        side_diag = np.ones(N-1)
        side_diag[np.arange(1,N)%nx==0] = 0
        up_down_diag = np.ones(N-nx)
        if bc == "periodic":
            side_loops = np.zeros(N-(nx-1))
            side_loops[np.arange(0,N-(nx-1))%nx==0] = 1
            up_loops = np.ones(nx)
            diagonals = [main_diag,side_diag,side_diag,side_loops,side_loops,up_down_diag,up_down_diag,up_loops,up_loops]
            offsets = [0, -1, 1,(nx-1),-(nx-1),(nx),-(nx),(N-nx),-(N-nx)]
        elif bc == "neumann":
            up_side = side_diag.copy()
            dn_side = side_diag.copy()
            up_neu = up_down_diag.copy()
            up_side[np.arange(0,N-1)%nx==0] = 2
            dn_side[np.arange(2,N+1)%nx==0] = 2
            up_neu[np.arange(0,(N-nx))//nx==0] = 2
            dn_neu = up_neu[::-1]
            diagonals = [main_diag,dn_side,up_side,up_neu,dn_neu]
            offsets = [0, -1, 1,(nx),-(nx)]
    gradient = diags(diagonals, offsets, format="csr")
    if len(coeff.shape)==1:
        blocks = [gradient*coeff[i] for i in xrange(coeff.shape[0])]
        gradient = block_diag(blocks)
    elif len(coeff.shape)==2:
        blocks=[]
        for i in range(coeff.shape[0]):
            row=[]
            for j in range(coeff.shape[1]):
                row.append(gradient*coeff[i,j])
            blocks.append(row)
        gradient=bmat(blocks)
    else:
        gradient=gradient*coeff[0]
    return csr_matrix(gradient/dx)

#if __name__ == '__main__':
#    import matplotlib.pyplot as plt
#    plt.matshow(create_laplacian_skelton((16,),bc="periodic").todense())
#    lap = create_laplacian((5,5),(2,2),coeff=[1.0,2.0])
#    print "Neumann -"
#    print lap.toarray()
##    plt.figure()
#    plt.matshow(lap.toarray())
#    plt.show()
#    lap = create_laplacian((5,5),(2,2),coeff=[[1.0,2.0],[2.0,1]])
#    print "Cross diffusion -"
#    print lap.toarray()
##    plt.figure()
#    plt.matshow(lap.toarray())
#    plt.show()

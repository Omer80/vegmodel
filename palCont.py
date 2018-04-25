#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:30:36 2018

Using Pseudo-arclength continuation scheme
https://math.stackexchange.com/questions/1415898/pseudo-arclength-continuation-scheme
Pseudo code:
given (u0,h0)
calculate Gu, Gh in (u0,h0)
calculate du0
take h1 = h0 + dh
initial guess:
    ug = u0 + du0*dh
solve:
    G(u,h1)=0
return u1
given (u1,h1)
...

To do :
    1) Standardize the way analytical and numerical functions work
    2) When numeric jacobians are chosen, the right hand side will not
       use any symbolic computation, but directly the right hand side
       of the equation

@author: omertz@post.bgu.ac.il
"""
from __future__ import print_function
from __future__ import division
__version__= 1.0
__author__ = """Omer Tzuk (omertz@post.bgu.ac.il)"""
#import time
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import gmres
import scipy.optimize as opt
from sympy import lambdify,symbols
import numdifftools as nd
from scipy.optimize import newton_krylov as solve

class ODECont(object):
    def __init__(self,model,parname,u0,h0,ds,analytic=True):
        """ model is the object of the model under study, parname is the
        symbol of the parameter of continuation. u0 is the initial value for
        the solution, and h0 is the initial value for the continuation
        parameter. ds is the arclength
        """
        self.m  = model
        self.h_ar = np.array([h0])
        self.u_ar = np.array([u0])
        self.pt_type_ar = np.array(["EP"])
        self.parname=parname
        self.u_0 = u0
        self.h_0 = h0
        self.pt_type = "EP"
        self.u_n = u0
        self.h_n = h0
        self.direction = np.sign(ds)
        self.ds  = np.fabs(ds)
        self.symh=symbols(parname)
        self.sym_uh = self.m.Vs_symbols[:]
        self.sym_uh.append(self.symh)
        self.symG  = self.m.symeqs
        self.symGuh= self.m.symeqs.jacobian(self.sym_uh)
        self.symGu = self.m.symeqs.jacobian(self.sym_uh[:-1])
        self.symGh = self.m.symeqs.jacobian([self.symh])
        # This is the rhs of the system of equations
        self.G     = lambdify(tuple(self.sym_uh),self.sub_parms(self.symG),"numpy",dummify=False)
#        if analytic==False:
        self.set_numeric_jacobians()
#        else:
        self.set_analytic_jacobians()
        self.du_n  = self.calcdu0()
        self.dh_n  = self.calcdh0()
        self.u_0    = opt.root(self.eqrhs,u0,args=(self.h_0)).x

    def sub_parms(self,eqs):
        for key in list(self.m.p.keys()):
            if key!=self.parname:
                eqs=eqs.subs(self.m.Ps_symbols[key],self.m.p[key])
        return eqs
    def eqrhs(self,u,h):
        if type(u)!=list:
            u = u.tolist()
        u.append(h)
        return np.squeeze(self.G(*u))
    def eqrhs_u(self,u):
        return np.squeeze(self.G(*u))
    def eqrhs_uh(self,uh):
        return np.squeeze(self.G(*uh))
    def set_numeric_jacobians(self):
        self.calcnumGuh=nd.Jacobian(self.eqrhs_uh)
        self.numGu=self.calcnumGu
        self.Guh=self.calcnumGuh
        self.Gu=self.calcnumGu
        self.Gh=self.calcnumGh
    def set_analytic_jacobians(self):
        # This is the Jacobian of the system together with derivation
        # in respect to the continuation parameter - with one extra
        # column
        self.anaGuh   = lambdify(tuple(self.sym_uh),self.sub_parms(self.symGuh),"numpy",dummify=False)
        # This is the normal jacobian of the system
        self.anaGu    = lambdify(tuple(self.sym_uh),self.sub_parms(self.symGu),"numpy",dummify=False)
        # This is a column vector of the derivation of rhs in respect to
        # the continuation parameter
        self.anaGh    = lambdify(tuple(self.sym_uh),self.sub_parms(self.symGh),"numpy",dummify=False)
        self.Guh=self.calcanaGuh
        self.Gu=self.calcanaGu
        self.Gh=self.calcanaGh
    def calcanaGuh(self,uh):
        if type(uh)!=list:
            uh = uh.tolist()
        return self.anaGuh(*uh)
    def calcanaGu(self,uh):
        if type(uh)!=list:
            uh = uh.tolist()
        return self.anaGu(*uh)
    def calcanaGh(self,uh):
        if type(uh)!=list:
            uh = uh.tolist()
        return self.anaGh(*uh)
    def calcnumGu(self,uh):
        return self.calcnumGuh(uh)[:,:-1]
    def calcnumGh(self,uh):
        return ((self.calcnumGuh(uh)[:,-1])[np.newaxis]).T
    def append_pt(self,u,h,pt_type):
        self.h_ar=np.append(self.h_ar,h)
        self.u_ar=np.vstack((self.u_ar,u))
        self.pt_type_ar=np.append(self.pt_type_ar,pt_type)
    def merge_uh(self,u,h):
        uh = list(u)
        uh.append(h)
        return uh
    def calcInvGuGh(self):
        u0h0 = self.merge_uh(self.u_0,self.h_0)
        return np.matmul(np.linalg.inv(self.Gu(u0h0)),np.squeeze(self.Gh(u0h0)))
    def calcdh0(self):
        invGuGh=self.calcInvGuGh()
        normGuGh = np.matmul(invGuGh,invGuGh)
        return self.direction*1.0/np.sqrt(normGuGh+1)
    def calcdu0(self):
        return np.squeeze(-1.0*self.calcInvGuGh()*self.calcdh0())
    def calcGuess(self):
        ug  = self.u_n  + self.du_n*self.ds
        hg  = self.h_n  + self.dh_n*self.ds
        return ug,hg
    def calcPerpenVec(self,u,h):
        """ Calculate the constraint of perpendicularity """
        return np.matmul((u-self.u_n),self.du_n) + (h-self.h_n)*self.dh_n - self.ds
    def eqGperp(self,uh):
        u = uh[:-1]
        h = uh[-1]
        Gperp = list(self.eqrhs(u,h))
        perp = self.calcPerpenVec(u,h)
        Gperp.append(perp)
        return Gperp
    def reconverge(self):
        return opt.root(self.eqGperp,self.merge_uh(self.u_n,self.h_n)).x
    def step(self):
        uh = self.reconverge()
        self.u_n=uh[:-1]
        self.h_n=uh[-1]
        Guh=self.Guh(uh)
        du_n_dh_n=np.concatenate((self.du_n,np.array([self.dh_n])))
        Guh_norm=np.vstack((Guh,du_n_dh_n))
        cond = np.zeros(Guh_norm.shape[0])
        cond[-1]=1.0
        du_n_dh_n,info=gmres(Guh_norm,cond)
#        return du_n_dh_n
        self.du_n=du_n_dh_n[:-1]
        self.dh_n=du_n_dh_n[-1]
        self.append_pt(self.u_n,self.h_n,self.pt_type)
    def follow(self,steps):
        for i in range(steps-1):
            self.pt_type=""
            self.step()
            print("h={},L2(u)={}".format(self.h_n,np.linalg.norm(self.u_n)))
        self.pt_type="EP"
        self.step()
        print("End point")
        print("h={},L2(u)={}".format(self.h_n,np.linalg.norm(self.u_n)))

# I need to create a function self.G that excepts a vector of the state 
# plus the value of the continuation parameter
class ReacDiffCont(object):
    def __init__(self,model,parname,u0,h0,ds,analytic=False):
        """ model is the object of the model under study, parname is the
        symbol of the parameter of continuation. u0 is the initial value for
        the solution, and h0 is the initial value for the continuation
        parameter. ds is the arclength
        """
        self.m  = model
        self.h_ar = np.array([h0])
        self.u_ar = np.array([u0])
        self.pt_type_ar = np.array(["EP"])
        self.parname=parname
        self.u_0 = u0
        self.h_0 = h0
        self.pt_type = "EP"
        self.u_n = u0
        self.h_n = h0
        self.ds  = ds
        self.numjac_delta=0.00000001
        self.symh=symbols(parname)
        self.sym_uh = self.m.Vs_symbols[:]
        self.sym_uh.append(self.symh)
        self.symeqs  = self.m.symeqs
        self.create_pde_eqs()
    def sub_parms(self,eqs):
        for key in list(self.m.p.keys()):
            if key!=self.parname:
                eqs=eqs.subs(self.m.Ps_symbols[key],self.m.p[key])
        return eqs
    def create_pde_eqs(self):
        from sympy.utilities.autowrap import ufuncify
        from utilities.laplacian_sparse import create_laplacian
        self.lapmat=create_laplacian(self.m.setup['n'],self.m.setup['l'], self.m.setup['bc'] , self.m.p['diffusion'])
        self.reac = []
        for i,var in enumerate(self.sym_uh[:-1]):
            self.reac.append(ufuncify(self.sym_uh,[self.sub_parms(self.symeqs[i])]))
    def split_uh(self,uh):
        h=uh[-1]
        u=uh[:-1]
        return np.split(u,len(self.sym_uh)-1),h
    def merge_uh(self,u,h):
        return np.append(np.ravel(u),h)
    def G(self,uh):
        u,h=self.split_uh(uh)
        rhs = []
        for i,var in enumerate(self.sym_uh[:-1]):
            rhs.append(self.reac[i](*(u+[h])))
        reac = np.ravel(rhs)
        diff = self.lapmat*np.ravel(u)
        return reac+diff
    def Guh(self,uh):
        n = len(uh)
        jacobian = []
        for j in range(n):
            state_plus = np.copy(uh)
            state_minus = np.copy(uh)
            state_plus[j] = state_plus[j]+self.numjac_delta
            state_minus[j] = state_minus[j]-self.numjac_delta
            jacobian.append((self.G(state_plus)-self.G(state_minus))/(2.0*self.numjac_delta))
        return sparse.csc_matrix(np.array(jacobian).T)
    def Gu(self,uh):
        u,h=self.split_uh(uh)
        u = np.ravel(u)
        n = len(u)
        jacobian = []
        for j in range(n):
            state_plus = np.copy(u)
            state_minus = np.copy(u)
            state_plus[j] = state_plus[j]+self.numjac_delta
            state_minus[j] = state_minus[j]-self.numjac_delta
            uh_plus = self.merge_uh(state_plus,h)
            uh_minus = self.merge_uh(state_minus,h)
            jacobian.append((self.G(uh_plus)-self.G(uh_minus))/(2.0*self.numjac_delta))
        return sparse.csc_matrix(np.array(jacobian).T)
    def Gh(self,uh):
        uh_plus = np.copy(uh)
        uh_minus = np.copy(uh)
        uh_plus[-1] = uh_plus[-1]+self.numjac_delta
        uh_minus[-1] = uh_minus[-1]-self.numjac_delta
        Gh=(self.G(uh_plus)-self.G(uh_minus))/(2.0*self.numjac_delta)
        return sparse.csc_matrix(Gh[np.newaxis].T)
    def calcInvGuGh(self,u,h):
        u0h0 = self.merge_uh(u,h)
        return np.matmul(np.linalg.inv(self.Gu(u0h0)),np.squeeze(self.Gh(u0h0)))
    def calcdh0(self,u,h):
        invGuGh=self.calcInvGuGh(u,h)
        normGuGh = np.matmul(invGuGh,invGuGh)
        return 1.0/np.sqrt(normGuGh+1)
    def calcdu0(self,u,h):
        return np.squeeze(-1.0*self.calcInvGuGh(u,h)*self.calcdh0(u,h))
    def calcGuess(self):
        ug  = self.u_n  + self.du_n*self.ds
        hg  = self.h_n  + self.dh_n*self.ds
        return ug,hg
    def calcPerpenVec(self,u,h):
        """ Calculate the constraint of perpendicularity """
        return np.matmul((u-self.u_n),self.du_n) + (h-self.h_n)*self.dh_n - self.ds
    def eqGperp(self,uh):
        u = uh[:-1]
        h = uh[-1]
        Gperp = list(self.eqrhs(u,h))
        perp = self.calcPerpenVec(u,h)
        Gperp.append(perp)
        return Gperp
    def reconverge(self):
        return solve(self.eqGperp,self.merge_uh(self.u_n,self.h_n)).x
    def step(self):
        uh = self.reconverge()
        self.u_n=uh[:-1]
        self.h_n=uh[-1]
        Guh=self.Guh(uh)
        du_n_dh_n=np.concatenate((self.du_n,np.array([self.dh_n])))
        Guh_norm=np.vstack((Guh,du_n_dh_n))
        cond = np.zeros(Guh_norm.shape[0])
        cond[-1]=1.0
        du_n_dh_n,info=gmres(Guh_norm,cond)
#        return du_n_dh_n
        self.du_n=du_n_dh_n[:-1]
        self.dh_n=du_n_dh_n[-1]
        self.append_pt(self.u_n,self.h_n,self.pt_type)
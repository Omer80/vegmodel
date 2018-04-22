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
@author: omertz@post.bgu.ac.il
"""
from __future__ import print_function
from __future__ import division
__version__= 1.0
__author__ = """Omer Tzuk (omertz@post.bgu.ac.il)"""
import time
import numpy as np
from scipy.sparse.linalg import gmres
import scipy.optimize as opt
from sympy import lambdify,symbols
#from scipy.optimize import newton_krylov as solve

class odeContinuation(object):
    def __init__(self,model,parname,u0,h0,ds,analytic=False):
        """ model is the object of the model under study, symh is the 
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
        symh=symbols(parname)
        self.sym_uh = self.m.Vs_symbols[:]
        self.sym_uh.append(symh)
        self.symG  = self.m.symeqs
        self.symGuh= self.m.symeqs.jacobian(self.sym_uh)
        self.symGu = self.m.symlocalJac
        self.symGh = self.m.symeqs.jacobian([symh])
        self.G     = lambdify(tuple(self.sym_uh),self.sub_parms(self.symG),"numpy",dummify=False)
        self.Guh   = lambdify(tuple(self.sym_uh),self.sub_parms(self.symGuh),"numpy",dummify=False)
        self.Gu    = lambdify(tuple(self.sym_uh),self.sub_parms(self.symGu),"numpy",dummify=False)
        self.Gh    = lambdify(tuple(self.sym_uh),self.sub_parms(self.symGh),"numpy",dummify=False)
        self.du_n  = self.calcdu0()
        self.dh_n  = self.calcdh0()
        self.u_0    = opt.root(self.eqrhs,[u0[0],u0[1]],args=(self.h_0)).x
        
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
        return np.matmul(np.linalg.inv(self.Gu(*u0h0)),np.squeeze(self.Gh(*u0h0)))
    def calcdh0(self):
        invGuGh=self.calcInvGuGh()
        normGuGh = np.matmul(invGuGh,invGuGh)
        return 1.0/np.sqrt(normGuGh+1)
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
        Guh=self.Guh(*uh)
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
            

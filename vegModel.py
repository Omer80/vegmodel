# -*- coding: utf-8 -*-
"""
#  vegModel.py
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
from __future__ import print_function
from __future__ import division
__version__= 1.0
__author__ = """Omer Tzuk (omertz@post.bgu.ac.il)"""
import time
from sympy import symbols, Matrix,lambdify
from sympy.utilities.autowrap import ufuncify
import numpy as np
import scipy.linalg as linalg
from scipy.integrate import odeint,ode
from scipy.optimize import newton_krylov
from scipy.optimize.nonlin import NoConvergence
from scipy.optimize import root as root_ode
from scipy.fftpack import fftn, ifftn
import scipy.sparse as sparse
from utilities import handle_netcdf as hn
import deepdish.io as dd

Es_1d    ={'rhs':"normal",
           'n':(1024,),
           'l':(256.0,),
           'bc':"periodic",
           'it':"pseudo_spectral",
           'dt':0.1,
           'analyze':True,
           'verbose':True,
           'setPDE':True}
Es_2d    ={'rhs':"normal",
           'n':(256,256),
           'l':(64.0,64.0),
           'bc':"neumann",
           'it':"rk4",
           'dt':0.1,
           'analyze':False,
           'verbose':True,
           'setPDE':True}

eta_rho =  [(3.421945725500000091e+00, 1.136291445400000079e-03),
            (3.412937193799999935e+00, 9.982219974600000170e-03),
            (3.323078563000000152e+00, 9.999999890100000344e-02),
            (3.227071132399999875e+00, 1.999432310200000007e-01),
            (3.090017596800000099e+00, 3.499496087000000255e-01),
            (2.999961518300000129e+00, 4.536235540799999910e-01),
            (2.321876860700000211e+00, 1.399952457499999969e+00),
            (1.999999998400000090e+00, 1.985519237899999956e+00),
            (1.992769139199999984e+00, 1.999970545100000052e+00),
            (1.556915407199999990e+00, 2.999998570799999875e+00),
            (1.216154001600000090e+00, 3.999993769200000049e+00),
            (9.384933219200000121e-01, 5.000000393399999687e+00),
            (7.047132087799999889e-01, 5.999995190399999956e+00),
            (5.025357171000000012e-01, 6.999985632499999610e+00),
            (3.237167218900000032e-01, 7.999984581899999725e+00),
            (1.624928677599999927e-01, 8.999990493900000388e+00),
            (1.470341472800000028e-02, 9.999995316399999723e+00)]

def main():
    global eta,rho
    Ps_eta = {'p':0.7,'l':1.428571429,'g':0.457142857,'a':0.0,
             'omegaf':0.5983986006837702,
             'e':eta_rho[0][0],'r':eta_rho[0][1],
             'diffusion':[1.0,150.0/1.2]}
    Ps_rho = {'p':0.7,'l':1.428571429,'g':0.457142857,'a':0.0,
             'omegaf':0.5983986006837702,
             'e':eta_rho[-1][0],'r':eta_rho[-1][1],
             'diffusion':[1.0,150.0/1.2]}
    eta = onfcModel(Ps_eta,Es_1d,None)
    rho = onfcModel(Ps_rho,Es_1d,None)
    return 0

class onfcModel(object):
    def __init__(self,Ps,Es,Vs=None):
        if type(Ps)==str:
            self.Psfname=Ps
            self.p=dd.load(Ps)
        else:
            self.Psfname=None
            self.p = Ps
        self.setup=Es
        self.setup['nvar']=2
#        self.Vs=Vs
        self.verbose=Es['verbose']
        if self.verbose:
            start=time.time()
        self.set_equations()
        self.dt = 0.1
        self.time_elapsed = 0
        if self.setup['setPDE']:
            self.p['nd']=len(Es['n'])
            if self.p['nd']==2:
                self.p['nx'],self.p['ny']=Es['n']
                self.p['lx'],self.p['ly']=Es['l']
                self.l=[self.p['lx'],self.p['ly']]
                self.n=[self.p['nx'],self.p['ny']]
                self.dg  = tuple([l/float(n) for l,n in zip(self.l,self.n)])
                self.dx  = self.dg[0]
            elif self.p['nd']==1:
                self.dg=[Es['l'][0]/float(Es['n'][0])]
                self.dx=self.dg[0]
            self.dx2 = self.dx**2
            self.dt=Es['dt']*self.dx2 / np.amax(self.p['diffusion'])
            self.X = np.linspace(0,Es['l'][0],Es['n'][0])
            from utilities.laplacian_sparse import create_laplacian #,create_gradient
            self.lapmat=create_laplacian(self.setup['n'],self.setup['l'], self.setup['bc'] , self.p['diffusion'],verbose=self.verbose)
            self.set_integrator()
            if self.verbose:
                print("Laplacian created")
        if Vs is not None:
            self.setup_initial_condition(Vs)
        if self.verbose:
            print("Time to setup: ",time.time()-start)
    """ Setting up model equations """
    def set_equations(self):
        b,w,t = symbols('b w t')
        self.Ps_symbols={}
        for key in list(self.p.keys()):
            self.Ps_symbols[key] = symbols(key)
        p=self.Ps_symbols
        if self.setup['rhs']=="normal":
            """ Normal mode """
            from sympy import cos as symcos
            p_t = p['p']*(1.0+p['a']*symcos(p['omegaf']*t))
            g   = (1.0+p['e']*b)*(1.0+p['e']*b)
            evap= (p['l']*w)/(1.0+p['r']*b)
            tras= p['g']*g*w*b
            self.dbdt_eq = w*b*g*(1.0-b)-b
            self.dwdt_eq = p_t-evap-tras
        if self.setup['rhs']=="cris":
            """ Normal mode """
            from sympy import cos as symcos
            p_t = p['p'] + p['a']*symcos(p['omegaf']*t)
            g   = (1.0+p['e']*b)*(1.0+p['e']*b)
            evap= (p['l']*w)/(1.0+p['r']*b)
            tras= p['g']*g*w*b
            self.dbdt_eq = w*b*g*(1.0-b)-b
            self.dwdt_eq = p_t-evap-tras
        """ Creating numpy functions """
        symeqs = Matrix([self.dbdt_eq,self.dwdt_eq])
        self.ode   = lambdify((b,w,t,p['p'],p['a'],p['omegaf']),self.sub_parms(symeqs),"numpy",dummify=False)
        self.dbdt  = ufuncify([b,w,t,p['p'],p['a'],p['omegaf']],[self.sub_parms(self.dbdt_eq)])
        self.dwdt  = ufuncify([b,w,t,p['p'],p['a'],p['omegaf']],[self.sub_parms(self.dwdt_eq)])
        localJac   = symeqs.jacobian(Matrix([b,w]))
        self.localJac = lambdify((b,w),self.sub_parms(localJac),"numpy",dummify=False)
        if self.setup['setPDE'] and self.setup['analyze']:
            self.dbdb = ufuncify([b,w],[self.sub_parms(localJac[0,0])])
            self.dbdw = ufuncify([b,w],[self.sub_parms(localJac[0,1])])
            self.dwdb = ufuncify([b,w],[self.sub_parms(localJac[1,0])])
            self.dwdw = ufuncify([b,w],[self.sub_parms(localJac[1,1])])
            k = symbols('k')
            delta_s  = symbols('delta_s')
            symeqs_lin_analysis = Matrix([self.dbdt_eq-b*k*k,self.dwdt_eq-w*delta_s*k*k])
            jaclinanalysis = symeqs_lin_analysis.jacobian(Matrix([b,w]))
            self.symbolic_jaclinanalysis = jaclinanalysis
            self.jaclinanalysis = lambdify((b,w,k),self.sub_parms(jaclinanalysis),"numpy",dummify=False)
        if self.verbose:
            self.print_equations()
            print("Local Jacobian:" ,localJac)
            if self.setup['setPDE'] and self.setup['analyze']:
                print("Linear analysis Jacobian: ", jaclinanalysis)

    """ Printing and parameters related functions """
    def print_parameters(self):
        print(self.p)
    def print_equations(self,numeric=False):
        if numeric:
            print("dbdt = ", self.sub_parms(self.dbdt_eq))
            print("dwdt  = ", self.sub_parms(self.dwdt_eq))
        else:
            print("dbdt = ", self.dbdt_eq)
            print("dwdt  = ", self.dwdt_eq)
    def print_latex_equations(self):
        from sympy import latex
        print("\partial_t b = ",latex(self.dbdt_eq))
        print("\partial_t w = ",latex(self.dwdt_eq))
        
    """ Functions for use with scipy methods """
    def calc_ode_eigs(self,state,t=0,**kwargs):
        b,w=state[0],state[1]
        if kwargs:
            self.update_parameters(kwargs)
        return linalg.eigvals(self.localJac(b,w))
    def sigma_k_scan(self,b,w,k_range=[0,1.0],n=1000,**kwargs):
        k_range = np.linspace(k_range[0],k_range[1],n)
        MaxReSigma = np.zeros(n)
        MaxImSigma = np.zeros(n)
        for i,k in enumerate(k_range):
            eigsvalues=linalg.eigvals(self.jaclinanalysis(b,w,k))
            MaxReSigma[i]=np.amax(np.real(eigsvalues))
            MaxImSigma[i]=np.imag(eigsvalues[np.argmax(np.real(eigsvalues))])
        return np.array([k_range,MaxReSigma,MaxImSigma])

    def calc_linear_stability_analysis(self,b,w,k_range=[0,1.0],n=1000):
        k_scan = self.sigma_k_scan(b,w,k_range=[0,0.1],n=1000)
        return k_scan[0][np.argmax(k_scan[1])],np.amax(k_scan[1])
    
    """ Utilities """
    def sub_parms(self,eqs):
        b,w,t = symbols('b w t')
        for key in list(self.p.keys()):
#            print key
            if key!='p' and key!='a' and key!='omegaf':
                eqs=eqs.subs(self.Ps_symbols[key],self.p[key])
        return eqs
    """ Spatial functions """
    def set_integrator(self):
        integrator_type = {}
        integrator_type['scipy'] = self.scipy_integrate
        integrator_type['euler'] = self.euler_integrate
        integrator_type['rk4'] = self.rk4_integrate
        integrator_type['pseudo_spectral'] = self.pseudo_spectral_integrate
        try:
            self.integrator = integrator_type[self.setup['it']]
        except KeyError:
            raise  ValueError("No such integrator : %s."%self.setup['it'])
        if self.setup['it']=='pseudo_spectral':
            self.dt*=100.0

    def rhs_pde(self,state,t=0):
        b,w=np.split(state,2)
        return np.ravel((self.dbdt(b,w,t,self.p['p'],self.p['a'],self.p['omegaf']),
                         self.dwdt(b,w,t,self.p['p'],self.p['a'],self.p['omegaf']))) + self.lapmat*state

    def rhs_ode(self,state,t=0):
        b,w=state
        return self.ode(b,w,t,self.p['p'],self.p['a'],self.p['omegaf']).T[0]
    def scipy_ode_rhs(self,t,state):
        b,w=state
        return self.ode(b,w,t,self.p['p'],self.p['a'],self.p['omegaf'])
    def scipy_ode_jac(self,t,state):
        b,w=state
        return self.localJac(b,w)
    def calc_pde_analytic_jacobian(self,state):
        b,w=np.split(state,2)
        dbdb=sparse.diags(self.dbdb(b,w))
        dbdw=sparse.diags(self.dbdw(b,w))
        dwdb=sparse.diags(self.dwdb(b,w))
        dwdw=sparse.diags(self.dwdw(b,w))
        local  = sparse.bmat([[dbdb,dbdw],
                              [dwdb,dwdw]])
        return sparse.csc_matrix(local)+sparse.csc_matrix(self.lapmat)

    def calc_ode_numerical_jacobian(self,b,w,delta=0.00000001):
        state = np.array([b,w])
        jacobian = []
        for j in range(len(state)):
            state_plus = np.copy(state)
            state_minus = np.copy(state)
            state_plus[j] = state_plus[j]+delta
            state_minus[j] = state_minus[j]-delta
            jacobian.append((np.array(self.dudt(state_plus))-np.array(self.dudt(state_minus)))/(2.0*delta))
        return np.array(jacobian).T
    def check_pde_jacobians(self,n=100):
        import time
        timer_analytic=0
        timer_numeric=0
        error = np.zeros(n)
        for i in range(n):
            print(i)
            b=np.random.random(self.setup['n'])
            w=np.random.random(self.setup['n'])
            state=np.ravel((b,w))
            start_time=time.time()
            numeric=self.calc_pde_numerical_jacobian(state)
            mid_time=time.time()
            analytic=self.calc_pde_analytic_jacobian(state)
            end_time=time.time()
            timer_numeric+=(mid_time-start_time)
            timer_analytic+=(end_time-mid_time)
            error[i]=np.max(np.abs(numeric-analytic))
        print("Max difference is ",np.max(error), ", and mean difference is ",np.mean(error))
        print("Average speed for numeric ", timer_numeric/float(n))
        print("Average speed for analytic ", timer_analytic/float(n))
        print("Analytic ", float(timer_numeric)/float(timer_analytic)," faster.")

    def calc_pde_numerical_jacobian(self,state,delta=0.00000001):
        n = len(state)
        jacobian = []
        for j in range(n):
            state_plus = np.copy(state)
            state_minus = np.copy(state)
            state_plus[j] = state_plus[j]+delta
            state_minus[j] = state_minus[j]-delta
            jacobian.append((self.rhs_pde(state_plus)-self.rhs_pde(state_minus))/(2.0*delta))
        return np.array(jacobian).T

    def calc_numeric_pde_eigs(self,state):
        return linalg.eigvals(self.calc_pde_numerical_jacobian(state))
    def calc_analytic_pde_eigs(self,state):
        return sparse.linalg.eigs(self.calc_pde_analytic_jacobian(state),k=3)[0]

    def check_convergence(self,state,previous_state,tolerance=1.0e-5):
        return np.max(np.abs(state-previous_state))<tolerance

    def update_parameters(self,parameters):
        intersection=[i for i in list(self.p.keys()) if i in parameters]
        if intersection:
            if self.setup['verbose']:
                print("Updating parameters:")
            for key in intersection:
                if self.setup['verbose']:
                    print(str(key)+"="+str(parameters[key]))
                self.p[key]=parameters[key]
                
    def integrate(self,initial_state=None,step=10,
                  max_time = 1000,tol=1.0e-5,plot=False,savefile=None,
                  create_movie=False,check_convergence=True,plot_update=False,
                  sim_step=None,**kwargs):
        if kwargs:
            self.update_parameters(kwargs)
        self.filename = savefile
        if initial_state is None:
            initial_state = self.initial_state
        if plot_update:
            b,w=initial_state.reshape(self.setup['nvar'],*self.setup['n'])
            import matplotlib.pyplot as plt
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_ylim([-0.1,1.0])
            ax1.set_xlim([0,self.setup['l'][0]])
            ax1.set_ylabel(r'$b$')
            ax2.set_ylabel(r'$w$')
            line1, = ax1.plot(self.X,b,'g')
            line2, = ax2.plot(self.X,w,'b')
            plt.draw()
        self.time_elapsed=0
        if sim_step is None:
            self.sim_step=0
        else:
            self.sim_step=sim_step
        if savefile is not None:
            hn.setup_simulation(savefile,self.p,self.setup)
            hn.save_sim_snapshot(savefile,self.sim_step,self.time_elapsed,
                                 self.split_state(initial_state),self.setup)
#        old_result = initial_state
        converged=False
        result = []
        result.append(initial_state)
#        t,result = self.integrator(initial_state,p=p,step=10,finish=10,savefile=self.filename)
        if self.setup['verbose']:
            start=time.time()
            print("Step {:4d}, Time = {:5.1f}".format(self.sim_step,self.time_elapsed))
        while not converged and self.time_elapsed<=max_time:
            old_result = result[-1]
            if plot_update:
                b,w=old_result.reshape(self.setup['nvar'],*self.setup['n'])
                line1.set_ydata(b)
                line2.set_ydata(w)
                fig.canvas.draw()
            t,result = self.integrator(result[-1],step=step,finish=step)
#            self.time_elapsed+=t[-1]
            self.sim_step=self.sim_step+1
            if savefile is not None:
                hn.save_sim_snapshot(savefile,self.sim_step,self.time_elapsed,
                                     self.split_state(result[-1]),self.setup)            
            if self.setup['verbose']:
                print("Step {:4d}, Time = {:10.6f}, diff = {:7f}".format(self.sim_step,self.time_elapsed,np.max(np.abs(result[-1]-old_result))))
            if check_convergence:
                converged=self.check_convergence(result[-1],old_result,tol)
                if converged:
                    print("Convergence detected")
        if self.setup['verbose']:
            print("Integration time was {} s".format(time.time()-start))
        if savefile is not None and create_movie:
            print("Creating movie...")
            hn.create_animation_b(savefile)
        return result[-1]

    """ Integrators step functions """
    def scipy_integrate(self,initial_state,step=0.1,finish=1000):
        """ """
#        print "Integration using scipy odeint"
        t = np.arange(0,finish+step, step)
        self.time_elapsed+=finish
        return t,odeint(self.rhs_pde, initial_state, t)
    def ode_integrate_bdf(self,initial_state=None,step=0.1,finish=1000,
                          max_dt=10.0e-3,max_order=15,**kwargs):
        if kwargs:
            self.update_parameters(kwargs)
        r = ode(self.scipy_ode_rhs, self.scipy_ode_jac).set_integrator('lsoda',max_step=max_dt, method='bdf', max_order_s=max_order)
        r.set_initial_value(initial_state, 0)
        t = []
        result = []
        t.append(0)
        result.append(initial_state)
        while r.successful() and r.t < finish:
            t.append(r.t+step)
            result.append(r.integrate(r.t+step))
        return np.array(t),np.array(result).T
    def ode_integrate(self,initial_state,step=0.1,start=0,finish=1000,**kwargs):
        """ """
        if kwargs:
            self.update_parameters(kwargs)
        t = np.arange(start,finish+step, step)
        return t,odeint(self.rhs_ode,initial_state, t).T

    def euler_integrate(self,initial_state=None,step=0.1,finish=1000,**kwargs):
        """ """
        if kwargs:
            self.update_parameters(kwargs)
        if initial_state is None:
            initial_state=self.state            
        time = np.arange(0,finish+step,step)
        result=np.zeros((len(time),len(initial_state)))
        t=0
        result[0]=initial_state
        for i,tout in enumerate(time[1:]):
            old=result[i]
            while t < tout:
#                print "t=",t
                new=old+self.dt*self.rhs_pde(old)
                old=new
                t+=self.dt
            result[i+1]=old
        self.state=result[-1]
        return t,result
    def rk4_integrate(self,initial_state=None,step=0.1,finish=1000,**kwargs):
        """ """
#        print "Integration using rk4 step"
        if kwargs:
            self.update_parameters(kwargs)
        if initial_state is None:
            initial_state=self.state
        time = np.arange(0,finish+step,step)
        result=np.zeros((len(time),len(initial_state)))
        t=0
        result[0]=initial_state
        for i,tout in enumerate(time[1:]):
            old=result[i]
            while t < tout:
                k1=self.dt*self.rhs_pde(old,self.time_elapsed)
                k2=self.dt*self.rhs_pde(old+0.5*k1,self.time_elapsed)
                k3=self.dt*self.rhs_pde(old+0.5*k2,self.time_elapsed)
                k4=self.dt*self.rhs_pde(old+k3,self.time_elapsed)
                new=old+(1.0/6.0)*(k1+2.0*k2+2.0*k3+k4)
                old=new
                t+=self.dt
                self.time_elapsed+=self.dt
            result[i+1]=old
        self.state=result[-1]
        return time,result
    def pseudo_spectral_integrate(self,initial_state=None,step=0.1,finish=1000,**kwargs):
#        print "Integration using pseudo-spectral step"
        if kwargs:
            self.update_parameters(kwargs)
        if initial_state is None:
            initial_state=self.state            
        time = np.arange(0,finish+step,step)
        result=np.zeros((len(time),len(initial_state)))
        t=0
        result[0]=initial_state
        for i,tout in enumerate(time[1:]):
            b,w=result[i].reshape(self.setup['nvar'],*self.setup['n'])
            self.fftb=fftn(b)
            self.fftw=fftn(w)
            while t < tout:
                self.fftb  = self.multb*(self.fftb + self.dt*fftn(self.dbdt(b,w,t,self.p['p'],self.p['a'],self.p['omegaf'])))#.real
                self.fftw  = self.multw*(self.fftw + self.dt*fftn(self.dwdt(b,w,t,self.p['p'],self.p['a'],self.p['omegaf'])))#.real
                b = ifftn(self.fftb).real
                w = ifftn(self.fftw).real
                t+=self.dt
                self.time_elapsed+=self.dt
            result[i+1]=np.ravel((b,w))
        self.state=result[-1]
        return time,result

    def spectral_multiplier(self,dt):
        n=self.setup['n']
        nx=n[0]
        dx=self.dx
        # wave numbers
        k=2.0*np.pi*np.fft.fftfreq(nx,dx)
        if len(n)==1:
            k2=k**2
        if len(n)==2:
            k2=np.outer(k,np.ones(nx))**2
            k2+=np.outer(np.ones(nx),k)**2
        # multiplier
        self.multb = np.exp(-dt*self.p['diffusion'][0]*k2)
        self.multw = np.exp(-dt*self.p['diffusion'][1]*k2)
    """ Auxilary root finding functions """
    def ode_root(self,initial_state,**kwargs):
        """ """
        if kwargs:
            self.update_parameters(kwargs)
        sol = root_ode(self.rhs_ode, initial_state)
        return sol.x
    def pde_root(self,initial_state, fixiter=100,tol=6e-6,smaxiter=1000,**kwargs):
        """ """
        if kwargs:
            self.update_parameters(kwargs)
        try:
            sol = newton_krylov(self.rhs_pde, initial_state,iter=fixiter, method='lgmres', verbose=int(self.setup['verbose']),f_tol=tol,maxiter=max(fixiter+1,smaxiter))
            converged=True
        except NoConvergence:
            converged=False
            if self.setup['verbose']:
                print("No Convergence flag")
            sol=initial_state
        return sol,converged

    def setup_initial_condition(self,Vs,**kwargs):
        n = self.setup['n']
        if type(Vs)==str:
            if Vs == "random":
                b = np.random.random(n)*0.5 + 0.1
                w = np.random.random(n)*(0.1) + 0.05
            elif Vs == "stable":
                b = np.ones(n)*(self.p['p']-self.p['l'])/(self.p['p']+self.p['g'])
                w = np.ones(n)*(self.p['p']+self.p['g'])/(self.p['l']+self.p['g'])
            elif Vs == "vegetated":
                p = kwargs.get('p', None)
                if p==None:
                    p=self.p['p']
                else:
                    self.p['p']=p
                t,result=self.ode_integrate_bdf([0.9,0.2],p,a=0.0)
                b_0,w_0 = result.T[-1]
                b = np.ones(n)*b_0
                w = np.ones(n)*w_0
            elif Vs == "vegetated_perturbed":
                p = kwargs.get('p', None)
                pert = kwargs.get('pert', None)
                if p==None:
                    p=self.p['p']
                else:
                    self.p['p']=p
                t,result=self.ode_integrate_bdf([0.9,0.2],p,a=0.0)
                b_0,w_0 = result.T[-1]
                b = np.ones(n)*b_0 + (np.random.random(n)-0.5)*pert
                w = np.ones(n)*w_0 + (np.random.random(n)-0.5)*pert
            self.initial_state = np.ravel((b,w))
        else:
            self.initial_state = Vs
        self.state = self.initial_state
        if self.setup['it'] == 'pseudo_spectral' and self.setup['setPDE']:
            self.spectral_multiplier(self.dt)

    def setup_initial_condition_b(self,Vs,**kwargs):
        n = self.setup['n']
        if type(Vs)==str:
            if Vs == "random":
                b1 = np.random.random(n)*0.5 + 0.1
                b2 = np.random.random(n)*0.5 + 0.1
                w  = np.random.random(n)*(0.1) + 0.05
            if Vs == "bare":
                b1 = np.zeros(n)
                b2 = np.zeros(n)
                w  = np.random.random(n)*(0.1) + 0.05
            elif Vs == "tile":
                fields = kwargs.get('fields', None)
                b,w = np.split(fields,self.setup['nvar'])
                b1 = np.tile(b1,(self.setup['n'][0],1))
                b2 = np.tile(b2,(self.setup['n'][0],1))
                w  = np.tile(w,(self.setup['n'][0],1))                
            elif Vs == "half":
#                import matplotlib.pyplot as plt
#                fig,ax = plt.subplots(2,1)
                p = kwargs.get('p', None)
#                w = kwargs.get('w', None)
                if p==None:
                    p=self.p['p']
                else:
                    self.p['p']=p
                t,result=self.ode_integrate_bdf([0.0,0.9,0.2],p,a=0.0)
#                ax[0].plot(t,result[1],'g')
                b1_1,b2_1,w_1 = result.T[-1]
                t,result=self.ode_integrate_bdf([0.9,0.0,0.2],p,a=0.0)
#                ax[1].plot(t,result[0],'r:')
                b1_2,b2_2,w_2 = result.T[-1]
                b1 = np.ones(n)*b1_1
                b2 = np.ones(n)*b2_1
                w  = np.ones(n)*w_1
                half = int(self.setup['n'][0]/2)
#                width_n = int((float(w)/(2.0*self.setup['l'][0]))*self.setup['n'][0])
                b1[:half]=b1_2
                b2[:half]=b2_2
                w[:half]=w_2
#                plt.show()
            elif Vs == "b":
#                import matplotlib.pyplot as plt
#                fig,ax = plt.subplots(2,1)
                p = kwargs.get('p', None)
#                w = kwargs.get('w', None)
                if p==None:
                    p=self.p['p']
                else:
                    self.p['p']=p
                t,result=self.ode_integrate_bdf([0.0,0.0,0.2],p,a=0.0)
#                ax[0].plot(t,result[1],'g')
                b1_1,b2_1,w_1 = result.T[-1]
                t,result=self.ode_integrate_bdf([0.9,0.0,0.2],p,a=0.0)
#                ax[1].plot(t,result[0],'r:')
                b1_2,b2_2,w_2 = result.T[-1]
                b1 = np.ones(n)*b1_2
                b2 = np.ones(n)*b2_2
                w  = np.ones(n)*w_2
                onethird = int(self.setup['n'][0]/3.0)
                twothird = int(2.0*self.setup['n'][0]/3.0)
#                width_n = int((float(w)/(2.0*self.setup['l'][0]))*self.setup['n'][0])
                b1[onethird:twothird]=b1_1
                b2[onethird:twothird]=b2_1
                w[onethird:twothird]=w_1
#                plt.show()
            elif Vs == "b2gap":
#                import matplotlib.pyplot as plt
#                fig,ax = plt.subplots(2,1)
                p = kwargs.get('p', None)
#                w = kwargs.get('w', None)
                if p==None:
                    p=self.p['p']
                else:
                    self.p['p']=p
                t,result=self.ode_integrate_bdf([0.0,0.0,0.2],p,a=0.0)
#                ax[0].plot(t,result[1],'g')
                b1_1,b2_1,w_1 = result.T[-1]
                t,result=self.ode_integrate_bdf([0.0,0.9,0.2],p,a=0.0)
#                ax[1].plot(t,result[0],'r:')
                b1_2,b2_2,w_2 = result.T[-1]
                b1 = np.ones(n)*b1_2
                b2 = np.ones(n)*b2_2
                w  = np.ones(n)*w_2
                onethird = int(self.setup['n'][0]/3.0)
                twothird = int(2.0*self.setup['n'][0]/3.0)
#                width_n = int((float(w)/(2.0*self.setup['l'][0]))*self.setup['n'][0])
                b1[onethird:twothird]=b1_1
                b2[onethird:twothird]=b2_1
                w[onethird:twothird]=w_1
#                plt.show()
            elif Vs == "b1gapb2stripe":
#                import matplotlib.pyplot as plt
#                fig,ax = plt.subplots(2,1)
                p = kwargs.get('p', None)
#                w = kwargs.get('w', None)
                if p==None:
                    p=self.p['p']
                else:
                    self.p['p']=p
                t,result=self.ode_integrate_bdf([0.0,0.9,0.2],p,a=0.0)
#                ax[0].plot(t,result[1],'g')
                b1_1,b2_1,w_1 = result.T[-1]
                t,result=self.ode_integrate_bdf([0.9,0.0,0.2],p,a=0.0)
#                ax[1].plot(t,result[0],'r:')
                b1_2,b2_2,w_2 = result.T[-1]
                b1 = np.ones(n)*b1_2
                b2 = np.ones(n)*b2_2
                w  = np.ones(n)*w_2
                onethird = int(self.setup['n'][0]/3.0)
                twothird = int(2.0*self.setup['n'][0]/3.0)
#                width_n = int((float(w)/(2.0*self.setup['l'][0]))*self.setup['n'][0])
                b1[onethird:twothird]=b1_1
                b2[onethird:twothird]=b2_1
                w[onethird:twothird]=w_1
#                plt.show()
            elif Vs == "b1stripe":
#                import matplotlib.pyplot as plt
#                fig,ax = plt.subplots(2,1)
                p = kwargs.get('p', None)
#                w = kwargs.get('w', None)
                if p==None:
                    p=self.p['p']
                else:
                    self.p['p']=p
                t,result=self.ode_integrate_bdf([0.9,0.0,0.2],p,a=0.0)
#                ax[0].plot(t,result[1],'g')
                b1_1,b2_1,w_1 = result.T[-1]
                t,result=self.ode_integrate_bdf([0.0,0.0,0.2],p,a=0.0)
#                ax[1].plot(t,result[0],'r:')
                b1_2,b2_2,w_2 = result.T[-1]
                b1 = np.ones(n)*b1_2
                b2 = np.ones(n)*b2_2
                w  = np.ones(n)*w_2
                onethird = int(self.setup['n'][0]/3.0)
                twothird = int(2.0*self.setup['n'][0]/3.0)
#                width_n = int((float(w)/(2.0*self.setup['l'][0]))*self.setup['n'][0])
                b1[onethird:twothird]=b1_1
                b2[onethird:twothird]=b2_1
                w[onethird:twothird]=w_1
#                plt.show()
            elif Vs == "b2stripe":
#                import matplotlib.pyplot as plt
#                fig,ax = plt.subplots(2,1)
                p = kwargs.get('p', None)
#                w = kwargs.get('w', None)
                if p==None:
                    p=self.p['p']
                else:
                    self.p['p']=p
                t,result=self.ode_integrate_bdf([0.0,0.9,0.2],p,a=0.0)
#                ax[0].plot(t,result[1],'g')
                b1_1,b2_1,w_1 = result.T[-1]
                t,result=self.ode_integrate_bdf([0.0,0.0,0.2],p,a=0.0)
#                ax[1].plot(t,result[0],'r:')
                b1_2,b2_2,w_2 = result.T[-1]
                b1 = np.ones(n)*b1_2
                b2 = np.ones(n)*b2_2
                w  = np.ones(n)*w_2
                onethird = int(self.setup['n'][0]/3.0)
                twothird = int(2.0*self.setup['n'][0]/3.0)
#                width_n = int((float(w)/(2.0*self.setup['l'][0]))*self.setup['n'][0])
                b1[onethird:twothird]=b1_1
                b2[onethird:twothird]=b2_1
                w[onethird:twothird]=w_1
#                plt.show()
            elif Vs == "halfb1":
#                import matplotlib.pyplot as plt
#                fig,ax = plt.subplots(2,1)
                p = kwargs.get('p', None)
#                w = kwargs.get('w', None)
                if p==None:
                    p=self.p['p']
                else:
                    self.p['p']=p
                t,result=self.ode_integrate_bdf([0.0,0.0,0.2],p,a=0.0)
#                ax[0].plot(t,result[1],'g')
                b1_1,b2_1,w_1 = result.T[-1]
                t,result=self.ode_integrate_bdf([0.9,0.0,0.2],p,a=0.0)
#                ax[1].plot(t,result[0],'r:')
                b1_2,b2_2,w_2 = result.T[-1]
                b1 = np.ones(n)*b1_1
                b2 = np.ones(n)*b2_1
                w  = np.ones(n)*w_1
                half = int(self.setup['n'][0]/2)
#                width_n = int((float(w)/(2.0*self.setup['l'][0]))*self.setup['n'][0])
                b1[:half]=b1_2
                b2[:half]=b2_2
                w[:half]=w_2
#                plt.show()
            elif Vs == "halfb2":
#                import matplotlib.pyplot as plt
#                fig,ax = plt.subplots(2,1)
                p = kwargs.get('p', None)
#                w = kwargs.get('w', None)
                if p==None:
                    p=self.p['p']
                else:
                    self.p['p']=p
                t,result=self.ode_integrate_bdf([0.0,0.9,0.2],p,a=0.0)
#                ax[0].plot(t,result[1],'g')
                b1_1,b2_1,w_1 = result.T[-1]
                t,result=self.ode_integrate_bdf([0.0,0.0,0.2],p,a=0.0)
#                ax[1].plot(t,result[0],'r:')
                b1_2,b2_2,w_2 = result.T[-1]
                b1 = np.ones(n)*b1_1
                b2 = np.ones(n)*b2_1
                w  = np.ones(n)*w_1
                half = int(self.setup['n'][0]/2)
#                width_n = int((float(w)/(2.0*self.setup['l'][0]))*self.setup['n'][0])
                b1[:half]=b1_2
                b2[:half]=b2_2
                w[:half]=w_2
#                plt.show()
            elif Vs == "halfrandom":
#                import matplotlib.pyplot as plt
                p = kwargs.get('p', None)
#                w = kwargs.get('w', None)
                if p==None:
                    p=self.p['p']
                else:
                    self.p['p']=p
                t,result=self.integrate_ode_bdf([0.01,0.2,0.2],p)
#                plt.plot(t,result[0])
                b_0,s1_0,s2_0 = result.T[-1]
                t,result=self.integrate_ode_bdf([0.9,0.2,0.2],p)
#                plt.plot(t,result[0])
                b_s,s1_s,s2_s = result.T[-1]
                b = np.ones(n)*b_0
                s1= np.ones(n)*s1_0
                s2= np.ones(n)*s2_0
                half = int(self.setup['n'][0]/2)
#                width_n = int((float(w)/(2.0*self.setup['l'][0]))*self.setup['n'][0])
                b[:half]=b_s*np.random.random(n)[:half]
                s1[:half]=s1_s*np.random.random(n)[:half]
                s2[:half]=s2_s*np.random.random(n)[:half]
#                plt.show()
            self.initial_state = np.ravel((b,w))
        else:
            self.initial_state = Vs
        self.state = self.initial_state
        if self.setup['it'] == 'pseudo_spectral' and self.setup['setPDE']:
            self.spectral_multiplier(self.dt)
    """ Plot functions """
    def plotLastFrame(self,initial_state=None,p=None,chi=None,beta=None,theta=None,savefile=None):
        sol=self.integrate_till_convergence(initial_state,p,chi,beta,theta,savefile)
        self.plot(sol)
        return sol

    def plot_ode(self,initial_state,step=0.1,finish=20,**kwargs):
        """ """
        if kwargs:
            self.update_parameters(kwargs)
        import matplotlib.pylab as plt
        t = np.arange(0,finish+step, step)
        sol=odeint(self.rhs_ode,initial_state, t).T
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(t,sol[0])
        ax2.plot(t,sol[1])
        plt.show()
        return t,sol
        
    def split_state(self,state):
        return state.reshape(self.setup['nvar'],*self.setup['n'])
    def split_sol(self,sol):
        return np.split(sol,self.setup['nvar'],axis=1)
    def save_pde_sol(self,fname,sol,t=None):
        b,w = self.split_sol(sol)
        data = {}
        data['b']=b
        data['w']=w
        data['Ps']=self.p
        data['Es']=self.setup
        if t is not None:
            data['t']=t
        dd.save(fname+'.hdf5',data,compression='blosc')
    
    def plot(self,state=None,fontsize=12,update=False,returner=False):
        if state is None:
            state=self.state
        import matplotlib.pylab as plt
        if update:
            plt.ion()
        b,w=state.reshape(self.setup['nvar'],*self.setup['n'])
        if len(self.setup['n'])==1:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_ylim([-0.1,1.0])
            ax1.set_xlim([0,self.setup['l'][0]])
            ax1.plot(self.X,b,'g')
            ax2.plot(self.X,w,'b')
            ax1.set_ylabel(r'$b$')
            ax2.set_ylabel(r'$w$')
        elif len(self.setup['n'])==2:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
            fig.subplots_adjust(right=0.8)
            bmin = np.amin(b)
            bmax = np.amax(b)
            ax1.imshow(b,cmap=plt.cm.YlGn, vmin = bmin, vmax = bmax)
            ax1.set_adjustable('box-forced')
            ax1.autoscale(False)
            ax1.set_title(r'$b$', fontsize=fontsize)
#            ax3.imshow(s1,cmap=plt.cm.Blues, vmin = smin, vmax = smax)
            ax2.imshow(w,cmap=plt.cm.Blues)
            ax2.set_adjustable('box-forced')
            ax2.autoscale(False)
            ax2.set_title(r'$w$', fontsize=fontsize)
#            cbar_ax2 = fig.add_axes([0.85, 0.54, 0.03, 0.35])
#            fig.colorbar(im1, cax=cbar_ax2)
#            plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
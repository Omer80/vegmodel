# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:31:01 2016

@author: Omer Tzuk <omertz@post.bgu.ac.il>
"""
import time as t
import netCDF4
import numpy as np


def setup_simulation(fname,Ps,Es):
    """
    Opening an netCDF4 file with 2 variables: b,w
    Also two groups are created:
    1 Ps - parameters - contains model's parameters set
    2 setup - contains model's setup switches, in particular 'nd' is the 
    number of dimensions
    value
    """
    with netCDF4.Dataset("%s.nc"%fname, 'w', format='NETCDF4') as rootgrp:
        print "Configuring netCDF4 file."
        setup = rootgrp.createGroup('setup')
        parms = rootgrp.createGroup('Ps')
        setattr(setup, 'nd',int(len(Es['n']))) 
        setattr(setup, 'nx',Es['n'][0]) 
        setattr(setup, 'lx',Es['l'][0])
        if int(len(Es['n']))==2:
            setattr(setup, 'ny',Es['n'][1]) 
            setattr(setup, 'ly',Es['l'][1])
        for k,v in Ps.items():
            if k!='dimpar':
                setattr(parms,k,v)
        rootgrp.description = "Simulation dataset for onfc2s model."
        rootgrp.history = "Created " + t.ctime(t.time())
        rootgrp.createDimension("x", Es['n'][0])
        rootgrp.createDimension('time', None)
        time = rootgrp.createVariable('time', 'f8', ('time',),zlib=True)
        time.units = "year"
        x = rootgrp.createVariable('x', 'f4', ('x',),zlib=True)
        x.units = "m"
        x[:] = np.linspace(0,Es['l'][0], Es['n'][0])
        if len(Es['n']) == 1:
            print "Setting up 1D variables"
            rootgrp.createVariable('b', 'f8', ('time', 'x',),zlib=True)
            rootgrp.createVariable('w', 'f8', ('time', 'x', ),zlib=True)
        elif len(Es['n']) == 2:
            print "Setting up 2D variables"
            rootgrp.createDimension("y", Es['n'][1])
            y = rootgrp.createVariable('y', 'f4', ('y',),zlib=True)
            y.units = "m"
            y[:] = np.linspace(0,Es['l'][1], Es['n'][1])
            rootgrp.createVariable('b', 'f8', ('time', 'x', 'y',),zlib=True)
            rootgrp.createVariable('w', 'f8',  ('time', 'x', 'y',),zlib=True)
        print "Output: netCDF file was created: ", fname+".nc"
    
def save_sim_snapshot(fname,step,time,Vs,Es):
    """ Save snapshot of the four fields b,w together with the time
    """
    b,w = Vs[0],Vs[1]
    with netCDF4.Dataset("%s.nc"%fname, 'a') as rootgrp:
        rootgrp['time'][step] = time
        if len(Es['n']) == 1:
            rootgrp['b'][step,:] = b
            rootgrp['w'][step,:]  = w
        elif len(Es['n']) == 2:
            rootgrp['b'][step,:,:] = b
            rootgrp['w'][step,:,:]  = w


def create_animation_b(fname,showtime=False):
    import matplotlib
    matplotlib.use('Agg')    
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    # row and column sharing
    fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
    ims=[]
    with netCDF4.Dataset("%s.nc"%fname, 'r', format='NETCDF4') as rootgrp:
        nd = int(getattr(rootgrp['setup'],'nd'))
        t = rootgrp['time'][:]
        if nd == 1:
    #        axes = plt.gca()
            ax.set_ylim([0.0,1.0])
            ax.set_ylabel(r'$b$', fontsize=25)
            ax.set_xlabel(r'$x$', fontsize=25)
            x  = rootgrp['x'][:]
            ax.set_xlim([x[0],x[-1]])
            b = rootgrp['b'][:,:]
            for i in xrange(len(t)):
                line, = ax.plot(x,b[i],'g-')
                if showtime:
                    ax.set_title(r'$b$ at $t={:4.3f}$'.format(t[i]), fontsize=25)
                ims.append([line])
        elif nd == 2:
            fig.subplots_adjust(right=0.8)
            ax.set_aspect('equal', 'datalim')
            ax.set_adjustable('box-forced')
    #        ax1.autoscale(False)
            ax.set_title(r'$b$', fontsize=25)
            b = rootgrp['b'][:,:,:]
            for i in xrange(len(t)):
                im = ax.imshow(b[i],cmap=plt.cm.YlGn, animated=True,vmin=0.0,vmax=1.0)
                if showtime:
                    ax.set_title(r'$b$ at $t={:4.3f}$'.format(t[i]), fontsize=25)
                ims.append([im])
                cbar_ax2 = fig.add_axes([0.85, 0.35, 0.05, 0.55])
                fig.colorbar(im, cax=cbar_ax2)
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save('%s.mp4'%fname)
    print "Movie made!"


def last_state(fname):
    with netCDF4.Dataset("%s.nc"%fname, 'r', format='NETCDF4') as rootgrp:
        nd = int(getattr(rootgrp['setup'],'nd'))
        if nd == 1:
            b = rootgrp['b'][:,-1]
            w = rootgrp['w'][:,-1]
        elif nd==2:
            b = rootgrp['b'][:,:,-1]
            w = rootgrp['w'][:,:,-1]
    return b,w

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:32:15 2020

@author: ben
"""
import numpy as np
import scipy.sparse as sp
from LSsurf.hermite_poly_fit import hermite_design_matrix

def scan_design_matrix(x, y, t, az, t0=None, nadir=False):
    d2r=np.pi/180
    if t0 is None:
        t0=np.linspace(np.nanmin(t), np.nanmax(t), 20)
    gi = hermite_design_matrix(t, t0)
    r,c,v=sp.find(gi)
    nri, nci=gi.shape
    G=sp.coo_matrix((nri*2, nci*6))
    # Center positions in x:
    G += sp.coo_matrix((v, (r,c)), shape=G.shape)
    # Center positions in y:
    G += sp.coo_matrix((v, (r+nri,c+nci)), shape=G.shape)
    if nadir:
        return G, None, t0
    
    # in-phase scan offset in x:
    G += sp.coo_matrix((v*np.cos(az[r]*d2r), (r, c+2*nci)), shape=G.shape)
    # quad-phase scan offset in x
    G += sp.coo_matrix((v*np.sin(az[r]*d2r), (r, c+3*nci)), shape=G.shape)
       
    # in-phase scan offset in y:
    G += sp.coo_matrix((v*np.cos(az[r]*d2r), (r+nri, c+4*nci)), shape=G.shape)
    # quad-phase scan offset in y
    G += sp.coo_matrix((v*np.sin(az[r]*d2r), (r+nri, c+5*nci)), shape=G.shape)
    # Constraints on the second differences of the model components
    nrc=t0.size-2
    ncc=t0.size
    Gc=sp.coo_matrix((6*nrc, 6*ncc))
    Gci=sp.diags([np.zeros(nrc)+0.01, np.zeros(nrc)+-0.02, np.zeros(nrc)+0.01], \
                 [0, 1, 2], shape=[nrc, ncc])
    r, c, v=sp.find(Gci)
    for ii in range(0, 6):
        Gc += sp.coo_matrix((v, (r+nrc*ii, c+ncc*ii)), shape=Gc.shape)
    return G, Gc, t0

def eval_ATM_scan(x, y, t, az, t0, xy_nadir=None, phase_xy=None, m=None, nadir=False):
    if m is None:
        if nadir:
            m=np.concatenate([xy_nadir.T, np.zeros(4, t0.size)], axis=0)
        m=np.c_[xy_nadir, phase_xy].T
    else:
        m=m.copy().T
    G=scan_design_matrix(x, y, t, az, t0, nadir=nadir)[0]
    bad=(G.power(2).dot(np.ones((G.shape[1],1)))==0).reshape(2, x.size).T
    xy_scan=G.dot(m.ravel().reshape(m.size, 1))
    xy_scan=xy_scan.reshape((2,x.size)).T
    xy_scan[bad]=np.NaN

    return xy_scan
    
def calc_xt_offset(x, y, t, az, t0, xy_nadir=None, phase_xy=None, m=None, nadir=False):
    """
    Calculate the along-track and across-track offsets for an ATM scan
    
    Parameters
    ----------
    x, y, t, az: np.arrays, [nData,]
        location, time, and scan azimuths for ATM footprints.  Should be 
        reduced from the full scan resolution
    t0 : np.array, nt0 points, optional
        times at which to evaluate the nadir location and scan information.
        If not specified, 20 points will be chosen spanning t
    xy_nadir : np.array, [2,nt0]
        Estimated nadir locations at t0
    amp : np.array, [nt0,]
        Estimated scan amplitude at t0
    phase_xy : np.array, [nt0x4]
        Estimated scan phase at t0
    t0 : np.array, [nt0,]
        timing of location estimates (calculated if not specified as an input)
    m : np.array, [nt0, 6]
        Combined array describing xy_nadir and phase_xy
    
    returns
    ------
    AT_offset, XT_offset: np.arrays, [nData,] 
        Along-track and across-track scan offsets at data points
    x_nadir, y_nadir: np.arrays, [N_data,]
        nadir locations for data points
    """
    
    
    if m is None:
        m=np.c_[xy_nadir, phase_xy].T
    else:
        m=m.copy().T
    G_nadir=scan_design_matrix(x, y, t, az, t0, nadir=True)[0]
    xy_nadir=G_nadir.dot(m.ravel().reshape((m.size,1)))
    xy_nadir=xy_nadir.reshape((2,x.size)).T
    vel=np.diff(xy_nadir, axis=0)
    vel=np.concatenate([vel, vel[-1,:].reshape(1,2)], axis=0)
    vel=vel.dot(np.array([1, 1j]).reshape(2,1))
    vel_hat=vel/np.abs(vel)
    
    #xy_offnadir=G[2:].dot(m[2:].ravel().reshape([t0.size*4, 1])).reshape(x.size,2)
    xy_offnadir=np.c_[x,y]-xy_nadir
    xy_offnadir=xy_offnadir.dot(np.array([1, 1j]).reshape(2,1))
    
    delta_xy=(np.conj(vel_hat)*xy_offnadir).ravel()
    return np.real(delta_xy), np.imag(delta_xy), xy_nadir[:,0].ravel(), xy_nadir[:,1].ravel()


def fix_time_res(days_J2k):
    ii=np.flatnonzero(np.diff(days_J2k)>0)
    t0=days_J2k[ii]
    nt=days_J2k.size
    # if the first good time difference happens after the start of the time seq, extrap backwards 
    # 
    if ii[0]>0:
        dtdi=(t0[1]-t0[0])/(ii[1]-ii[0])
        ii=np.concatenate([[0], ii])
        t0=np.concatenate([[t0[0]-ii[0]*dtdi], t0])
    if ii[-1]< nt:
        dtdi=(t0[-1]-t0[-2])/(ii[-1]-ii[-2])
        di=nt-ii[-1]
        ii=np.concatenate([ii, [nt]])
        t0=np.concatenate([t0, [t0[-1]+dtdi*di]])
    return np.interp(np.arange(nt), ii, t0)

def fit_ATM_data(D, calc_fit=False):
    """
    Fit scan parameters to an ATM data structure
    
    Parameters
    ----------
    D : pointCollection.data
        ATM scanning laser data containing fields:
            x, y, days_J2k, scan_az

    Returns
    -------
    D : pointCollection.data
        ATM data updated with scan_AT, scan_XT, scan_amp, x_nadir and y_nadir fields

    """
    t=fix_time_res(D.days_J2k)
    n_secs=(np.nanmax(t)-np.nanmin(t))*24*3600
    t0=np.linspace(np.min(t),np.max(t),np.max([20, np.ceil(n_secs/2)]).astype('i'))
    
    good=np.flatnonzero(np.isfinite(D.x+D.y+t) & (np.abs(D.x+1j*D.y) < 1.e7))
    good=good[::50]
    xy_naidr0, amp, phase_xy, t0, m = fit_ATM_scan(
        D.x[good], D.y[good], t[good], D.azimuth[good], t0=t0)
    x_off, y_off, x_nadir, y_nadir=calc_xt_offset(D.x, D.y,t, D.azimuth, t0, m=m)
    scan_amp=np.interp(t, t0, amp)
    D.assign({'scan_AT':x_off, 'scan_XT':y_off, 
              'x_nadir': x_nadir.ravel(), 
              'y_nadir': y_nadir.ravel(),
              'scan_amp':scan_amp.ravel()})
    if calc_fit:
        xy_est=eval_ATM_scan(D.x, D.y, t, D.azimuth, t0, m=m)
        D.assign({'x_fit':xy_est[:,0].ravel(), 'y_fit':xy_est[:,1].ravel()})
    return D

def fit_ATM_scan(x, y, t, az, t0=None):
    """
    Estimate the nadir location and scan parameters for ATM data

    Parameters
    ----------
    x, y, t, az: np.arrays, nData points
        location, time, and scan azimuths for ATM footprints.  Should be 
        reduced from the full scan resolution
 
    t0 : np.array, nt0 points, optional
        times at which to evaluate the nadir location and scan information.
        If not specified, 20 points will be chosen spanning t

    Returns
    -------
    xy_naidr : np.array, [2,nt0]
        Estimated nadir locations
    amp : np.array, [nt0,]
        Estimated scan amplitude
    phase_xy : np.array, [nt0x4]
        Estimated scan phase
    t0 : np.array, [nt0,]
        timing of location estimates (calculated if not specified as an input)
    m : np.array, [nt0, 6]
        Combined array describing xy_nadir and phase_xy
        

    """
    if t0 is None:
        t0=np.linspace(np.nanmin(t), np.nanmax(t), 20)
    Gd, Gc=scan_design_matrix(x, y, t, az, t0)[0:2]
    d=np.concatenate([x.ravel(), y.ravel(), np.zeros(Gc.shape[0])])
    G=sp.vstack([Gd, Gc]).tocsr()
    m=sp.linalg.spsolve(G.T.dot(G), G.T.dot(d.reshape((d.size,1))))
    m=m.reshape((6, t0.size)).T
    
    xy_nadir=m[:, 0:1]
    amp=np.sqrt(np.sum(m[:,2:]**2, axis=1))
    phase_xy=m[:, 2:]
    return xy_nadir, amp, phase_xy, t0, m

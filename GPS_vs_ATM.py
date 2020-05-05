# -*- coding: utf-8 -*-
"""
Compare ATM data with GPS data.

This script calculates differences between GPS points and the ATM data that
measured the same surface region.

Created on Wed Sep  5 13:36:08 2018

@author: ben
"""

import pointCollection as pc
#from PointDatabase import ATL06_filters
from ATL11.RDE import RDE
import numpy as np
import pickle
import h5py
import os
import re
import sys
import argparse
#from ATL11.pt_blockmedian import pt_blockmedian
from PointDatabase import pt_blockmedian
from ATM_waveform.fit_ATM_scan import fit_ATM_data
from sklearn.neighbors import KDTree

# WGS84 semimajor and semiminor axes
WGS84a=6378137.0
WGS84b=6356752.31424
d2r=np.pi/180.

def my_lsfit(G, d):
    try:
        # this version of the inversion is much faster than linalg.lstsq
        m=np.linalg.solve(G.T.dot(G), G.T.dot(d))#, rcond=None)
        #m=m0[0]
    except ValueError:
        print("ValueError in LSq")
        return np.NaN+np.zeros(G.shape[1]), np.NaN, np.NaN
    except np.linalg.LinAlgError:
        print("LinalgError in LSq")
        return np.NaN+np.zeros(G.shape[1]), np.NaN, np.NaN
    r=d-G.dot(m)
    R=np.sqrt(np.sum(r**2)/(d.size-G.shape[1]))
    sigma_hat=RDE(r)
    return m, R, sigma_hat

def blockmedian_for_gps(GPS, delta):
    # make a subset of the GPS data that contains the blockmedian elevation values

    lat0=np.nanmedian(GPS.latitude)
    lon0=np.nanmedian(GPS.longitude)
    # calculate the ellipsoid radius for the current point
    Re=WGS84a**2/np.sqrt((WGS84a*np.cos(d2r*lat0))**2+(WGS84b*np.sin(d2r*lat0))**2)
    delta_lon=np.mod(GPS.longitude-lon0+180.,360.)-180
    # project the GPS latitude and longitude into northing and easting
    EN=Re*np.c_[delta_lon*np.cos(d2r*lat0), (GPS.latitude-lat0)]*np.pi/180.
    z=GPS.z.astype(np.float64)
    xm, ym, zm, ind=pt_blockmedian(EN[:,0], EN[:,1], z, delta, return_index=True);#, randomize_evens=True)

    for field in GPS.fields:
        temp=getattr(GPS, field)
        setattr(GPS, field, 0.5*(temp[ind[:,0]]+temp[ind[:,1]]))
    GPS.longitude=0.5*(delta_lon[ind[:,0]]+delta_lon[ind[:,1]])+lon0
    return GPS

def blockmedian_for_qsub(Qdata, delta):
    # make a subset of the qfit data that contains the blockmedian elevation values

    lat0=np.nanmedian(Qdata.latitude)
    lon0=np.nanmedian(Qdata.longitude)
    # calculate the ellipsoid radius for the current point
    Re=WGS84a**2/np.sqrt((WGS84a*np.cos(d2r*lat0))**2+(WGS84b*np.sin(d2r*lat0))**2)
    delta_lon=np.mod(Qdata.longitude-lon0+180.,360.)-180
    # project the Qfit latitude and longitude into northing and easting
    EN=Re*np.c_[delta_lon*np.cos(d2r*lat0), (Qdata.latitude-lat0)]*np.pi/180.
    Qz=Qdata.elevation.astype(np.float64)
    xm, ym, zm, ind=pt_blockmedian(EN[:,0], EN[:,1], Qz, delta, return_index=True);#, randomize_evens=True)

    for field in Qdata.fields:
        temp=getattr(Qdata, field)
        setattr(Qdata, field, 0.5*(temp[ind[:,0]]+temp[ind[:,1]]))
    Qdata.longitude=0.5*(delta_lon[ind[:,0]]+delta_lon[ind[:,1]])+lon0
    return Qdata

def compare_gps_with_qfit(GPSsub, Qdata, out_template):

    # calculate the ellipsoid radius for the current point
    lat0=GPSsub.latitude
    lon0=GPSsub.longitude
    Re=WGS84a**2/np.sqrt((WGS84a*np.cos(d2r*lat0))**2+(WGS84b*np.sin(d2r*lat0))**2)

    # project the Qfit latitude and longitude into northing and easting
    EN=Re*np.c_[(np.mod(Qdata.longitude-lon0+180.,360.)-180)*np.cos(d2r*lat0), (Qdata.latitude-lat0)]*np.pi/180.

    # take a 50-meter circular subset
    ind_50m=np.sum(EN**2,axis=1)<50**2
    if np.sum(ind_50m) < 10:
        return None
    EN=EN[ind_50m,:]
    # elevation of Qfit data
    z=Qdata.elevation[ind_50m].astype(np.float64)
    # cross track position
    if 'scan_XT' in Qdata.fields:
        scan_XT=Qdata.scan_XT[ind_50m]

    # convert into local polar stereo as a function of distance from point
    xy_local = np.c_[Qdata.x[ind_50m] - GPSsub.x, Qdata.y[ind_50m] - GPSsub.y]

    # copy the GPS values into the output template
    this_out=out_template.copy()
    copy_fields=['longitude','latitude','x','y','z']

    for field in copy_fields:
        this_out[field]=getattr(GPSsub, field)

    #fit a plane to all data within 50m of the point
    G=np.c_[np.ones((ind_50m.sum(), 1)), xy_local]
    if not np.all(np.isfinite(G.ravel())):
        print("Bad G")
    try:
        m, R, sigma_hat=my_lsfit(G, z)
    except TypeError:
        return None
    this_out['sigma_qfit_50m']=R
    this_out['h_qfit_50m']=m[0]
    this_out['dh_qfit_dx']=m[1]
    this_out['dh_qfit_dy']=m[2]
    this_out['N_50m']=np.sum(ind_50m)
    this_out['dz_50m'],=np.array(GPSsub.z-m[0],dtype=np.float64)
    this_out['RDE_50m']=sigma_hat
    this_out['t_qfit']=np.nanmean(Qdata.days_J2k[ind_50m])
    # if running the scan program
    if 'scan_XT' in Qdata.fields:
        this_out['scan_XT_50m']=np.nanmean(scan_XT)

    ind_20m=np.sum(EN**2,axis=1)<20**2
    if (np.sum(ind_20m) > 5):
        this_out['hbar_20m']=np.nanmean(z[ind_20m])

    sub_10m=np.logical_and(np.abs(xy_local[:,1])<10, np.abs(xy_local[:,0])<10)
    if (np.sum(sub_10m) > 5):
        G=np.c_[np.ones((sub_10m.sum(), 1)), xy_local[sub_10m,:]]
        m, R, sigma_hat=my_lsfit(G, z[sub_10m])
        this_out['sigma_qfit_10m']=R
        this_out['h_qfit_10m']=m[0]
        this_out['N_10m']=np.sum(sub_10m)
        this_out['dz_10m'],=np.array(GPSsub.z-m[0],dtype=np.float64)
        this_out['RDE_10m']=sigma_hat
        this_out['y_10m_mean']=np.nanmean(xy_local[sub_10m,1])
        this_out['x_10m_mean']=np.nanmean(xy_local[sub_10m,0])
        # if running the scan program
        if 'scan_XT' in Qdata.fields:
            this_out['scan_XT_10m']=np.nanmean(scan_XT[sub_10m])

    # return the output dictionary
    return this_out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gps','-G', type=str, help="GPS file to run")
    parser.add_argument('--atm', '-A', type=str,  help='ATM directory to run')
    parser.add_argument('--hemisphere','-H', type=int, default=-1, help='hemisphere, must be 1 or -1')
    parser.add_argument('--query','-Q', type=float, default=100, help='KD-Tree query radius')
    parser.add_argument('--median','-M', default=False, action='store_true', help='Run block median')
    parser.add_argument('--scan','-S', default=False, action='store_true', help='Run ATM scan fit')
    parser.add_argument('--verbose','-v', default=False, action='store_true', help='verbose output of run')
    args=parser.parse_args()

    if args.hemisphere==1:
        SRS_proj4 = '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs '
    elif args.hemisphere==-1:
        SRS_proj4 = '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

    # tilde expansion of file arguments
    GPS_file=os.path.expanduser(args.gps)
    fileBasename, fileExtension = os.path.splitext(GPS_file)
    ATM_dir=os.path.expanduser(args.atm)

    print("working on GPS file {0}, ATM directory {1}".format(GPS_file, ATM_dir)) if args.verbose else None

    # find Qfit files within ATM_dir
    Qfit_regex = re.compile(r"ATM1B.*_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2}).*.h5$")
    Qfit_files = [os.path.join(ATM_dir,f) for f in os.listdir(ATM_dir) if Qfit_regex.search(f)]

    # output directory
    out_dir=os.path.join(ATM_dir,'xovers')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # output file
    out_file = 'vs_{0}.h5'.format(os.path.basename(fileBasename))
    # check if output file exists
    if os.path.isfile(os.path.join(out_dir,out_file)):
        print("found: {0}".format(os.path.join(out_dir,out_file))) if args.verbose else None

    # read GPS HDF5 file
    GPS_field_dict = {None:['latitude','longitude','z']}
    GPS=pc.data().from_h5(GPS_file,field_dict=GPS_field_dict).get_xy(SRS_proj4)
    # run block median over GPS data
    if args.median:
        GPS=blockmedian_for_gps(GPS, 5)

    # read all Qfit files within ATM directory
    Qlist=list()
    for f in sorted(Qfit_files):
        Qlist.append(pc.ATM_Qfit.data().from_h5(f))
    # merge the list of ATM data and build the search tree
    Q_full=pc.data().from_list(Qlist).get_xy(SRS_proj4)

    # fit scan parameters to an ATM data structure
    if args.scan:
        Q_full=fit_ATM_data(Q_full)

    # run block median for qsub
    if args.median:
        Q_full=blockmedian_for_qsub(Q_full, 5)

    # construct search tree from ATM Qfit coords
    # pickle Qtree to save computational time for future runs
    if os.path.isfile(os.path.join(ATM_dir,'tree.p')):
        Qtree = pickle.load(open(os.path.join(ATM_dir,'tree.p'),'rb'))
    else:
        Qtree = KDTree(np.c_[Q_full.x,Q_full.y])
        pickle.dump(Qtree,open(os.path.join(ATM_dir,'tree.p'),'wb'))

    # output fields
    out_fields=['x','y','z','longitude','latitude',
        't_qfit','h_qfit_50m','sigma_qfit_50m','dz_50m','RDE_50m','N_50m',
        'hbar_20m','h_qfit_10m','sigma_qfit_10m','dz_10m','RDE_10m','N_10m',
        'x_10m_mean','y_10m_mean']
    # append scan fields to output template
    if args.scan:
        out_fields.extend(['scan_XT_50m','scan_XT_10m'])
    out_template={f:np.NaN for f in out_fields}
    out=list()

    # query the search tree to find points within query radius
    Qquery = Qtree.query_radius(np.c_[GPS.x, GPS.y], args.query)
    # indices of GPS points within bin
    ind, = np.nonzero([np.any(i) for i in Qquery])
    # loop over queries in the GPS data
    for i in ind:
        GPSsub = GPS.copy_subset(np.array([i]))
        # grab the Qfit bins around the GPS bin
        Qdata = Q_full.copy_subset(Qquery[i], by_row=True)
        Qdata.index(np.isfinite(Qdata.elevation) & np.isfinite(Qdata.latitude) & np.isfinite(Qdata.longitude))
        # create output dictionary of GPS and plane-fit ATM comparison
        this_out = compare_gps_with_qfit(GPSsub, Qdata, out_template)
        if this_out is not None:
            out.append(this_out)

    # if there were overlapping points between the GPS and ATM data
    if out:
        D=dict()
        with h5py.File(os.path.join(out_dir,out_file),'w') as h5f:
            for field in out[0].keys():
                D[field]=np.array([ii[field] for ii in out])
                print(field,D[field].dtype) if args.verbose else None
                h5f.create_dataset(field, data=D[field])

# run main program
if __name__ == '__main__':
    main()

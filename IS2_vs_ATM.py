# -*- coding: utf-8 -*-
"""
Compare ATM data with ICESat-2 data.

This script uses the geo_index framework to load spatially overlapping blocks of
ATM and ATL06 data, and calculate differences between ATL06 segments and the
ATM data that measured the same surface patch.

Created on Wed Sep  5 13:36:08 2018

@author: ben
"""

import pointCollection as pc
#from PointDatabase import ATL06_filters
from ATL11.RDE import RDE
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import re
import sys
#from ATL11.pt_blockmedian import pt_blockmedian
from PointDatabase import pt_blockmedian
from ATM_waveform.fit_ATM_scan import fit_ATM_data
from sklearn.neighbors import KDTree

DOPLOT=False
VERBOSE=True

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

def compare_seg_with_qfit(D6i, Qdata, out_template):

    # calculate the ellipsoid radius for the current point
    lat0=D6i.latitude
    lon0=D6i.longitude
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
    scan_XT=Qdata.scan_XT[ind_50m]

    # calculate along-track vector and the across-track vector
    this_az=D6i.seg_azimuth[0]
    if not np.isfinite(this_az):
        return None
    at_vec=np.array([np.sin(this_az*d2r), np.cos(this_az*d2r)])
    xt_vec=at_vec[[1,0]]*np.array([-1, 1])

    # project the Qfit data into the along-track coordinate system
    xy_at=np.c_[np.dot(EN, at_vec), np.dot(EN, xt_vec)]

    # copy the ATL06 values into the output template
    this_out=out_template.copy()
    copy_fields=['segment_id','x','y', 'dh_fit_dx', 'h_li','h_li_sigma', 'h_mean','orbit_number',
                 'atl06_quality_summary', 'w_surface_window_final','n_fit_photons', 'fpb_n_corr',
                 'delta_time', 'h_robust_sprd', 'snr_significance','y_atc','rgt','dac','spot']

    for field in copy_fields:
        this_out[field]=getattr(D6i, field)

    #fit a plane to all data within 50m of the point
    G=np.c_[np.ones((ind_50m.sum(), 1)), xy_at]
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
    this_out['dz_50m']=D6i.h_li-m[0]
    this_out['RDE_50m']=sigma_hat
    this_out['scan_XT_50m']=np.nanmean(scan_XT)

    ind_20m=np.sum(EN**2,axis=1)<20**2
    if np.sum(ind_20m)> 5:
        this_out['hbar_20m']=np.nanmean(z[ind_20m])

    sub_seg=np.logical_and(np.abs(xy_at[:,1])<5, np.abs(xy_at[:,0])<30)
    if np.sum(sub_seg)<10:
        return None
    G=np.c_[np.ones((sub_seg.sum(), 1)), xy_at[sub_seg,0]]
    m, R, sigma_hat=my_lsfit(G, z[sub_seg])
    this_out['sigma_qfit_seg']=R
    this_out['h_qfit_seg']=m[0]
    this_out['N_seg']=np.sum(sub_seg)
    this_out['dz_seg']=D6i.h_li-m[0]
    this_out['RDE_seg']=sigma_hat
    this_out['beam_pair']=D6i.BP
    this_out['scan_XT_seg']=np.nanmean(scan_XT[sub_seg])
    this_out['t_qfit']=np.nanmean(Qdata.days_J2k[ind_50m])
    this_out['y_seg_mean']=np.nanmean(xy_at[sub_seg,1])
    this_out['x_seg_mean']=np.nanmean(xy_at[sub_seg,0])

    # return the output dictionary
    return this_out


sigma_pulse=5.5

ATL06_dir=sys.argv[1]
ATM_dir=os.path.expanduser(sys.argv[2])

print("working on ATL06 dir {0}, ATM directory {1}".format(ATL06_dir,  ATM_dir)) if VERBOSE else None

# find Qfit files within ATM_dir
Qfit_regex = re.compile(r"ATM1B.*_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2}).*.h5")
Qfit_files = [os.path.join(ATM_dir,f) for f in os.listdir(ATM_dir) if Qfit_regex.search(f)]

# output directory
out_dir=os.path.join(ATM_dir,'xovers')
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

out_file = 'vs_{0}.h5'.format(os.path.dirname(ATL06_dir).replace(os.sep, '_'))
if os.path.isfile(os.path.join(out_dir,out_file)):
    print("found: {0}".format(os.path.join(out_dir,out_file))) if VERBOSE else None

ATL06_index=os.path.join(os.sep,'Volumes','ice2','ben','scf','AA_06',ATL06_dir,'GeoIndex.h5')

SRS_proj4='+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
ATL06_field_dict={None:['delta_time','h_li','h_li_sigma','latitude','longitude','segment_id','sigma_geo_h','atl06_quality_summary'],
            'ground_track':['x_atc', 'y_atc','seg_azimuth','sigma_geo_at','sigma_geo_xt'],
            'geophysical':['dac'],
            'bias_correction':['fpb_n_corr'],
            'fit_statistics':['dh_fit_dx','dh_fit_dx_sigma','dh_fit_dy','h_mean', 'h_rms_misfit','h_robust_sprd','n_fit_photons','w_surface_window_final','snr_significance'],
            'orbit_info':['rgt','orbit_number'],
            'derived': ['BP','spot']}

ATL06_fields=list()
for key in ATL06_field_dict:
    ATL06_fields+=ATL06_field_dict[key]

# WGS84 semimajor and semiminor axes
WGS84a=6378137.0
WGS84b=6356752.31424
d2r=np.pi/180.
delta=[10000., 10000.]
# query radius for finding overlapping points
qradius = 100

# read all Qfit files within ATM directory
Qlist=list()
for f in sorted(Qfit_files):
    Qlist.append(pc.ATM_Qfit.data().from_h5(f))
# merge the list of ATM data and build the search tree
Q_full=pc.data().from_list(Qlist).get_xy(SRS_proj4)
# run block median for qsub
Q_full=blockmedian_for_qsub(Q_full, 5)
# fit scan parameters to an ATM data structure
Q_full=fit_ATM_data(Q_full)
# construct search tree from ATM Qfit coords
Qtree = KDTree(np.c_[Q_full.x,Q_full.y])
# could pickle Qtree here to save computational time for future runs

# read 10 km ATL06 index
D6_GI=pc.geoIndex(SRS_proj4=SRS_proj4).from_file(ATL06_index, read_file=True)
# Query the Qfit search tree to find intersecting ATL06 bins
# search within radius equal to diagonal of bin with 1km buffer (12/sqrt(2))
x_10km,y_10km = D6_GI.bins_as_array()
D6ind, = np.nonzero(Qtree.query_radius(np.c_[x_10km,y_10km],8485,count_only=True))
# reduce ATL06 bins to valid
D6_GI = D6_GI.copy_subset(xyBin=[x_10km[D6ind], y_10km[D6ind]])

out_fields=[
    'segment_id','x','y', 'BP', 'h_li', 'h_li_sigma', 'atl06_quality_summary',
    'dac', 'rgt','orbit_number','spot',
    'dh_fit_dx','N_50m','N_seg','h_qfit_seg','dh_qfit_dx','dh_qfit_dy',
    'h_robust_sprd', 'snr_significance',
    'h_qfit_50m','sigma_qfit_50m', 'sigma_seg','dz_50m','E_seg','RDE_seg',
    'scan_XT_50m','scan_XT_seg','hbar_20m',
    'RDE_50m','t_seg','t_qfit','y_atc', 'x_seg_mean', 'y_seg_mean']
out_template={f:np.NaN for f in out_fields}
out=list()


# plt.figure(1)
for bin_name in sorted(D6_GI.keys()):
    #plt.clf()
    print(bin_name) if VERBOSE else None
    bin_xy=[int(coord) for coord in bin_name.split('_')]

    # query ATL06 for the current bin, and index it
    D6list=D6_GI.query_xy([[bin_xy[0]], [bin_xy[1]]], get_data=True, fields=ATL06_field_dict)
    if not isinstance(D6list, list):
        D6list=[D6list]
    D6sub=pc.ATL06.data().from_list(D6list).get_xy(SRS_proj4)
    D6sub.ravel_fields()
    # query the search tree to find points within qradius
    #D6xy = np.c_[(np.nanmean(D6sub.x, axis=1),np.nanmean(D6sub.y, axis=1))]
    D6sub.index(np.isfinite(D6sub.x) & np.isfinite(D6sub.h_li))
    D6xy = np.c_[D6sub.x, D6sub.y]
    Qquery = Qtree.query_radius(D6xy, qradius)
    # indices of ATL06 points within bin
    D6ind, = np.nonzero([np.any(i) for i in Qquery])
    # loop over queries in the ATL06 data
    for i_AT in D6ind:
        D6i = D6sub.copy_subset(np.array([i_AT]))
        # grab the Qfit bins around the ATL06 bin
        Qdata = Q_full.copy_subset(Qquery[i_AT], by_row=True)
        Qdata.index(np.isfinite(Qdata.elevation) & np.isfinite(Qdata.latitude) & np.isfinite(Qdata.longitude))
        # create output dictionary of ATL06 and plane-fit ATM comparison
        this_out = compare_seg_with_qfit(D6i, Qdata, out_template)
        if this_out is not None:
            out.append(this_out)

D=dict()
with h5py.File(os.path.join(out_dir,out_file),'w') as h5f:
    for field in out[0].keys():
        D[field]=np.array([ii[field] for ii in out])
        h5f.create_dataset(field, data=D[field])

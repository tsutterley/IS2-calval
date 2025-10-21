# -*- coding: utf-8 -*-
"""
Compare GPS data with ICESat-2 data.

This script calculates differences between GPS points and the ATL06 data that
measured the same surface patch.

Created on Wed Sep  5 13:36:08 2018

@author: ben
"""

import os
import re
import sys
import h5py
import pickle
import argparse
import numpy as np
import pointCollection as pc
from pointCollection import pt_blockmedian, RDE
from sklearn.neighbors import KDTree

# WGS84 semimajor and semiminor axes
WGS84a=6378137.0
WGS84b=6356752.31424
d2r=np.pi/180.
delta=[10000., 10000.]

def my_lsfit(G, d):
    try:
        # this version of the inversion is much faster than linalg.lstsq
        m=np.linalg.solve(G.T.dot(G), G.T.dot(d))#, rcond=None)
        #m=m0[0]
    except ValueError:
        print("ValueError in LSq")
        return np.nan+np.zeros(G.shape[1]), np.nan, np.nan
    except np.linalg.LinAlgError:
        print("LinalgError in LSq")
        return np.nan+np.zeros(G.shape[1]), np.nan, np.nan
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

def compare_seg_with_gps(D6i, GPS, out_template):

    # calculate the ellipsoid radius for the current point
    lat0=D6i.latitude
    lon0=D6i.longitude
    Re=WGS84a**2/np.sqrt((WGS84a*np.cos(d2r*lat0))**2+(WGS84b*np.sin(d2r*lat0))**2)

    # project the GPS latitude and longitude into northing and easting
    EN=Re*np.c_[(np.mod(GPS.longitude-lon0+180.,360.)-180)*np.cos(d2r*lat0), (GPS.latitude-lat0)]*np.pi/180.

    # take a 50-meter circular subset
    ind_50m=np.sum(EN**2,axis=1)<50**2
    if np.sum(ind_50m) < 10:
        return None
    EN=EN[ind_50m,:]
    # elevation of GPS data
    z=GPS.z[ind_50m].astype(np.float64)

    # calculate along-track vector and the across-track vector
    this_az=D6i.seg_azimuth[0]
    if not np.isfinite(this_az):
        return None
    at_vec=np.array([np.sin(this_az*d2r), np.cos(this_az*d2r)])
    xt_vec=at_vec[[1,0]]*np.array([-1, 1])

    # project the gps data into the along-track coordinate system
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
    this_out['sigma_gps_50m']=R
    this_out['h_gps_50m']=m[0]
    this_out['dh_gps_dx']=m[1]
    this_out['dh_gps_dy']=m[2]
    this_out['N_50m']=np.sum(ind_50m)
    this_out['dz_50m']=D6i.h_li-m[0]
    this_out['RDE_50m']=sigma_hat

    ind_20m=np.sum(EN**2,axis=1)<20**2
    if np.sum(ind_20m)> 5:
        this_out['hbar_20m']=np.nanmean(z[ind_20m])

    sub_seg=np.logical_and(np.abs(xy_at[:,1])<5, np.abs(xy_at[:,0])<30)
    if np.sum(sub_seg)<10:
        return None
    G=np.c_[np.ones((sub_seg.sum(), 1)), xy_at[sub_seg,0]]
    m, R, sigma_hat=my_lsfit(G, z[sub_seg])
    this_out['sigma_gps_seg']=R
    this_out['h_gps_seg']=m[0]
    this_out['N_seg']=np.sum(sub_seg)
    this_out['dz_seg']=D6i.h_li-m[0]
    this_out['RDE_seg']=sigma_hat
    this_out['beam_pair']=D6i.BP
    this_out['y_seg_mean']=np.nanmean(xy_at[sub_seg,1])
    this_out['x_seg_mean']=np.nanmean(xy_at[sub_seg,0])

    # return the output dictionary
    return this_out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gps','-G', type=str, help="GPS file to run")
    parser.add_argument('--atl06', '-I', type=str,  help='ICESat-2 ATL06 directory to run')
    parser.add_argument('--hemisphere','-H', type=int, default=-1, help='hemisphere, must be 1 or -1')
    parser.add_argument('--query','-Q', type=float, default=100, help='KD-Tree query radius')
    parser.add_argument('--median','-M', default=False, action='store_true', help='Run block median')
    parser.add_argument('--verbose','-v', default=False, action='store_true', help='verbose output of run')
    args=parser.parse_args()

    if args.hemisphere==1:
        SRS_proj4 = '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs '
        HEM = 'GL_06'
    elif args.hemisphere==-1:
        SRS_proj4 = '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
        HEM = 'AA_06'

    # tilde expansion of file arguments
    GPS_file=os.path.expanduser(args.gps)
    GPS_dir=os.path.dirname(GPS_file)

    print("working on ATL06 dir {0}, GPS file {1}".format(args.atl06,  GPS_file)) if args.verbose else None

    # output directory
    out_dir=os.path.join(GPS_dir,'xovers')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_file = 'vs_{0}.h5'.format(os.path.dirname(args.atl06).replace(os.sep, '_'))
    if os.path.isfile(os.path.join(out_dir,out_file)):
        print("found: {0}".format(os.path.join(out_dir,out_file))) if args.verbose else None

    ATL06_index=os.path.join(os.sep,'Volumes','ice2','ben','scf',HEM,args.atl06,'GeoIndex.h5')

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

    # read GPS HDF5 file
    GPS_field_dict = {None:['latitude','longitude','z']}
    GPS_full=pc.data().from_h5(GPS_file,field_dict=GPS_field_dict).get_xy(SRS_proj4)
    # run block median over GPS data
    if args.median:
        GPS_full=blockmedian_for_gps(GPS_full, 5)
    # construct search tree from GPS coordinates
    # pickle tree to save computational time for future runs
    if os.path.isfile(os.path.join(GPS_dir,'tree.p')):
        tree = pickle.load(open(os.path.join(GPS_dir,'tree.p'),'rb'))
    else:
        tree = KDTree(np.c_[GPS_full.x,GPS_full.y])
        pickle.dump(tree,open(os.path.join(GPS_dir,'tree.p'),'wb'))

    # read 10 km ATL06 index
    D6_GI=pc.geoIndex(SRS_proj4=SRS_proj4).from_file(ATL06_index, read_file=True)
    # Query the gps search tree to find intersecting ATL06 bins
    # search within radius equal to diagonal of bin with 1km buffer (12/sqrt(2))
    x_10km,y_10km = D6_GI.bins_as_array()
    D6ind, = np.nonzero(tree.query_radius(np.c_[x_10km,y_10km],8485,count_only=True))
    # reduce ATL06 bins to valid
    D6_GI = D6_GI.copy_subset(xyBin=[x_10km[D6ind], y_10km[D6ind]])

    out_fields=[
        'segment_id','x','y', 'BP', 'h_li', 'h_li_sigma', 'atl06_quality_summary',
        'dac', 'rgt','orbit_number','spot',
        'dh_fit_dx','N_50m','N_seg','h_gps_seg','dh_gps_dx','dh_gps_dy',
        'h_robust_sprd', 'snr_significance',
        'h_gps_50m','sigma_gps_50m', 'sigma_seg','dz_50m','E_seg','RDE_seg',
        'hbar_20m','RDE_50m','t_seg','y_atc', 'x_seg_mean', 'y_seg_mean']
    out_template={f:np.nan for f in out_fields}
    out=list()

    for bin_name in sorted(D6_GI.keys()):
        print(bin_name) if args.verbose else None
        bin_xy=[int(coord) for coord in bin_name.split('_')]

        # query ATL06 for the current bin, and index it
        D6list=D6_GI.query_xy([[bin_xy[0]], [bin_xy[1]]], get_data=True, fields=ATL06_field_dict)
        if not isinstance(D6list, list):
            D6list=[D6list]
        D6sub=pc.ATL06.data().from_list(D6list).get_xy(SRS_proj4)
        D6sub.ravel_fields()
        # query the search tree to find points within query radius
        #D6xy = np.c_[(np.nanmean(D6sub.x, axis=1),np.nanmean(D6sub.y, axis=1))]
        D6sub.index(np.isfinite(D6sub.x) & np.isfinite(D6sub.h_li))
        D6xy = np.c_[D6sub.x, D6sub.y]
        query = tree.query_radius(D6xy, args.query)
        # indices of ATL06 points within bin
        D6ind, = np.nonzero([np.any(i) for i in query])
        # loop over queries in the ATL06 data
        for i_AT in D6ind:
            D6i = D6sub.copy_subset(np.array([i_AT]))
            # grab the gps bins around the ATL06 bin
            GPS = GPS_full.copy_subset(query[i_AT], by_row=True)
            GPS.index(np.isfinite(GPS.z) & np.isfinite(GPS.latitude) & np.isfinite(GPS.longitude))
            # create output dictionary of ATL06 and GPS comparison
            this_out = compare_seg_with_gps(D6i, GPS, out_template)
            if this_out is not None:
                out.append(this_out)

    D=dict()
    with h5py.File(os.path.join(out_dir,out_file),'w') as h5f:
        for field in out[0].keys():
            D[field]=np.array([ii[field] for ii in out])
            print(field,D[field].dtype) if args.verbose else None
            h5f.create_dataset(field, data=D[field])

# run main program
if __name__ == '__main__':
    main()

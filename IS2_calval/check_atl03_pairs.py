#!/usr/bin/env python
"""
check_atl03_pairs.py
Written by Tyler Sutterley (01/2026)
Check differences between ICESat-2 ATL03 beam pair heights

UPDATE HISTORY:
    Written 01/2026
"""
import re
import pathlib
import argparse
import warnings
import numpy as np
import pandas as pd
import IS2_calval as is2cv
# ignore pandas warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Check differences between ICESat-2
            ATL03 beam pair heights
            """
    )
    # command line options
    parser.add_argument('infile',
        type=pathlib.Path, 
        help='ICESat-2 ATL03 file to run')
    parser.add_argument('--threshold', '-t',
        type=float, default=5,
        help='Maximum absolute value for DOT difference (meters)')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # granule name
    granule = args.infile.name
    # regular expression pattern for extracting information
    pattern = r'(ATL\d{2})_(\d{14})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2}).h5$'
    rx = re.compile(pattern, re.VERBOSE)
    PRD, YYYYMMDDHHMMSS, RGT, CYC, GRAN, RL, VERS = rx.findall(granule).pop()
    # minimum weight for reference photon selection
    minimum_weight = 200 if (int(RL) < 7) else 6000
    # additional variables to read
    field_mapping = dict(
        segment_id='geolocation/segment_id',
        podppd_flag='geolocation/podppd_flag'
    )
    # associated beam pairs
    pair_tracks = dict(gt1l='gt1r',  gt2l='gt2r', gt3l='gt3r')
    # read ATL03 file
    df = is2cv.io.read_granule(args.infile, field_mapping=field_mapping)
    # calculate orthometric heights for each beam
    df['h_ortho'] = np.nan
    for group in df.ground_track.unique():
        # reduce to ground track
        df1 = df[df.ground_track == group]
        # extract reference photon height
        df1['reference_photon_height'] = is2cv.io.reference_photon_height(
            args.infile, group, minimum_weight=minimum_weight)
        # reduce to surface type
        is_type = is2cv.io.is_surface_type(args.infile, group)
        df1.loc[~is_type, 'reference_photon_height'] = np.nan
        # compute dynamic ocean topographies
        df.loc[df.ground_track == group, 'h_ortho'] = \
            df1['reference_photon_height'] - (df1['geoid'] + df1['geoid_free2mean'])

    # list to hold dataframes
    dataframes = []
    # calculate difference between beam pairs
    for group, cmp in pair_tracks.items():
        # reduce to ground track
        df1 = df[df.ground_track == group]
        df2 = df[df.ground_track == cmp]
        # reduce to corresponding segment ids
        mask1 = df1.segment_id.isin(df2.segment_id)
        mask2 = df2.segment_id.isin(df1.segment_id)
        df1 = df1[mask1].set_index('segment_id')
        df2 = df2[mask2].set_index('segment_id')
        # if there is any overlap: compute height differences
        df1['dh_pair'] = np.abs(df1.h_ortho - df2.h_ortho)
        # check for outliers
        outliers = (df1['dh_pair'] > args.threshold) & (np.isfinite(df1['dh_pair']))
        dataframes.append(df1[outliers])
    # concatenate dataframes
    df1 = pd.concat(dataframes, ignore_index=False)
    # don't write out empty dataframes
    if df1.empty:
        return
    # write dataframe to output parquet file
    outfile = (f'{PRD}_PT_Cycle{CYC}_RGT{RGT}_{GRAN}_R{RL}_{VERS}.parquet')
    df1.to_parquet(args.infile.with_name(outfile), index=False)

if __name__ == '__main__':
    main()

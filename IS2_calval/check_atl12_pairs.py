#!/usr/bin/env python
"""
check_atl12_pairs.py
Written by Tyler Sutterley (10/2025)
Check differences between ICESat-2 ATL12 beam pair heights

UPDATE HISTORY:
    Written 10/2025
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
            ATL12 beam pair heights
            """
    )
    # command line options
    parser.add_argument('infile',
        type=pathlib.Path, 
        help='ICESat-2 ATL12 file to run')
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
    # additional variables to read
    field_mapping = dict(
        first_geoseg='ssh_segments/stats/first_geoseg',
        last_geoseg='ssh_segments/stats/last_geoseg',
        podppd_flag_seg='ssh_segments/stats/podppd_flag_seg'
    )
    # associated beam pairs
    associated_beam_pair = dict(
        gt1l='gt1r', gt1r='gt1l',
        gt2l='gt2r', gt2r='gt2l',
        gt3l='gt3r', gt3r='gt3l'
    )
    # read ATL12 file
    df = is2cv.io.read_granule(args.infile, field_mapping=field_mapping)
    # update RGT from orbit number
    df['track'] = is2cv.io.orbit_number_to_track(df['orbit_number'])
    # compute dynamic ocean topographies
    df['h_ortho'] = df['h'] - (df['geoid_seg'] + df['geoid_free2mean_seg'])
    # list to hold dataframes
    dataframes = []
    # calculate difference between beam pairs
    for group in df.ground_track.unique():
        # associated beam in pair
        cmp = associated_beam_pair[group]
        # reduce to ground track
        df1 = df[df.ground_track == group]
        df2 = df[df.ground_track == cmp]
        df1['dh_pair'] = np.nan
        # for each row in the dataframe
        for i, row in df1.iterrows():
            # find corresponding segment in paired beam
            mask = (df2.first_geoseg <= row.last_geoseg) & \
                   (df2.last_geoseg >= row.first_geoseg) & \
                   (df2.orbit_number == row.orbit_number)
            # if there is any overlap: compute height differences
            if np.any(mask):
                h_ortho_cmp = df2['h_ortho'][mask].mean()
                df1.loc[i, 'dh_pair'] = np.abs(row.h_ortho - h_ortho_cmp)
        # check for outliers
        outliers = (df1['dh_pair'] > args.threshold) & (np.isfinite(df1['dh_pair']))
        dataframes.append(df1[outliers])
    # concatenate dataframes
    df1 = pd.concat(dataframes, ignore_index=True)
    # write dataframe to output parquet file
    outfile = (f'{PRD}_PT_Cycle{CYC}_RGT{RGT}_R{RL}_{VERS}.parquet')
    df1.to_parquet(args.infile.with_name(outfile), index=False)
    df1.to_csv(args.infile.with_name(outfile.replace('.parquet','.csv')), index=False)

if __name__ == '__main__':
    main()

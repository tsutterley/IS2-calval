#!/usr/bin/env python
"""
check_atl12_dot.py
Written by Tyler Sutterley (10/2025)
Check ICESat-2 ATL12 dynamic ocean topography estimates

UPDATE HISTORY:
    Written 10/2025
"""
import re
import pathlib
import argparse
import numpy as np
import IS2_calval as is2cv

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Checks ICESat-2 ATL12 dynamic ocean
            topography (DOT) estimates
            """
    )
    # command line options
    parser.add_argument('infile',
        type=pathlib.Path, 
        help='ICESat-2 ATL12 file to run')
    parser.add_argument('--threshold', '-t',
        type=float, default=4,
        help='Maximum absolute value for DOT (meters)')
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
    # read ATL12 file
    df = is2cv.io.read_granule(args.infile, field_mapping=field_mapping)
    # update RGT from orbit number
    df['track'] = is2cv.io.orbit_number_to_track(df['orbit_number'])
    # compute dynamic ocean topographies
    df['h_ortho'] = df['h'] - (df['geoid_seg'] + df['geoid_free2mean_seg'])
    # filter to values greater than threshold
    mask = (df['h_ortho'].abs() >= args.threshold)
    # skip if no data with extreme values
    if not np.any(mask):
        return
    # filter dataframe to bounds
    df1 = df.loc[mask,:].reset_index(drop=True)
    # add file-level attributes
    # standard deviation
    df1.attrs['stdev'] = float(df['h_ortho'].std())
    # robust dispersion estimate
    P16 = df['h_ortho'].quantile(0.16)
    P84 = df['h_ortho'].quantile(0.84)
    df1.attrs['RDE'] = float((P84 - P16)/2.0)
    # add spot-level attributes
    for spot in df.atlas_spot_number.unique():
        # attributes for spot
        group = f'spot{spot:d}'
        df1.attrs[group] = {}
        # reduce to spot
        df2 = df[df.atlas_spot_number == spot]
        # standard deviation
        df1.attrs[group]['stdev'] = float(df2['h_ortho'].std())
        # robust dispersion estimate
        P16 = df2['h_ortho'].quantile(0.16)
        P84 = df2['h_ortho'].quantile(0.84)
        df1.attrs[group]['RDE'] = float((P84 - P16)/2.0)
    # write dataframe to output parquet file
    outfile = (f'{PRD}_Cycle{CYC}_RGT{RGT}_R{RL}_{VERS}.parquet')
    df1.to_parquet(args.infile.with_name(outfile), index=False)

if __name__ == '__main__':
    main()

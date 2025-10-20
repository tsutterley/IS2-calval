#!/usr/bin/env python
"""
tech_ref_events.py
Written by Tyler Sutterley (10/2025)
Check ICESat-2 data before and after events in the Tech Ref Table

Tech Ref Table:
https://doi.org/10.5281/zenodo.16283560
"""
import re
import h5py
import argparse
import numpy as np
import pandas as pd
import icesat2_toolkit as is2tk
import utilities

# get path to data directory
filepath = utilities.get_data_path(['data'])

# variable mapping
mapping = {}
mapping['ATL12'] = dict(
    delta_time = 'ssh_segments/delta_time',
    longitude = 'ssh_segments/longitude',
    latitude = 'ssh_segments/latitude',
    fpb_corr = 'ssh_segments/heights/fpb_corr',
    h = 'ssh_segments/heights/h',
    h_ice_free = 'ssh_segments/heights/h_ice_free',
    h_ice_free_uncrtn = 'ssh_segments/heights/h_ice_free_uncrtn',
    geoid_free2mean_seg = 'ssh_segments/stats/geoid_free2mean_seg',
    geoid_seg = 'ssh_segments/stats/geoid_seg',
    ice_conc = 'ssh_segments/stats/ice_conc',
    n_photon = 'ssh_segments/stats/n_photons',
    n_ttl_photon = 'ssh_segments/stats/n_ttl_photon',
    near_sat_fract_seg = 'ssh_segments/stats/near_sat_fract_seg'
)

def from_excel(xls_file=None, pattern=r'OceanScan'):
    # get excel file if not provided
    if xls_file is None:
        # get zenodo url and checksum for the tech ref table
        zenodo_url, checksum = utilities.get_zenodo_url()
        # download excel file from zenodo
        # and validate against checksum
        xls_file = utilities.from_http(zenodo_url, hash=checksum)
    # get sheet names from excel file
    sheets = utilities.get_excel_sheet_names(xls_file,
        pattern=r'Cycle\s+(\d+)')
    # compile regex for filtering rows
    rx = re.compile(pattern, re.IGNORECASE)
    # output dict for dataframe
    data = {}
    data['delta_time_start'] = []
    data['delta_time_end'] = []
    data['rgt_start'] = []
    data['rgt_end'] = []
    data['orbit_start'] = []
    data['orbit_end'] = []
    data['cycle'] = []
    # for each sheet in the excel file
    for i, sheet_name in enumerate(sheets):
        df = pd.read_excel(xls_file, sheet_name=sheet_name, header=1)
        for j, row in df.iterrows():
            if rx.search(row['DETAILS']) and np.isfinite(row['BEG RGT']):
                # extract delta time from row
                time_beg = row['ATL03 DELTA_TIME START (seconds)']
                time_end = row['ATL03 DELTA_TIME END (seconds)']
                data['delta_time_start'].append(time_beg)
                data['delta_time_end'].append(time_end)
                # extract orbit number from row
                orbit_beg = int(row['BEG ORBIT'])
                orbit_end = int(row['END ORBIT'])
                data['orbit_start'].append(orbit_beg)
                data['orbit_end'].append(orbit_end)
                # extract RGT from row
                rgt_beg = int(row['BEG RGT'])
                rgt_end = int(row['END RGT'])
                data['rgt_start'].append(rgt_beg)
                data['rgt_end'].append(rgt_end)
                # extract cycle number from sheet name
                cycle, = re.findall(r'Cycle\s+(\d+)', sheet_name, re.I)
                data['cycle'].append(int(cycle))
    # convert to pandas dataframe
    df = pd.DataFrame(data)
    return df

def all_tracks(df):
    # verify dataframe is iterable
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    # build lists with all cycles and RGTs
    cycles, tracks = [], []
    # for each row in the dataframe
    for row in df.itertuples():
        # list of RGTs for row
        RGT = [r for r in range(int(row.rgt_start), int(row.rgt_end)+1)]
        # cycle as string
        cycle = [int(row.cycle)]*len(RGT)
        # extend output lists
        cycles.extend(cycle)
        tracks.extend(RGT)
    # convert to dataframe
    df1 = pd.DataFrame({'cycle':cycles, 'track':tracks})
    # reduce to unique values
    df1 = df1.drop_duplicates().reset_index(drop=True)
    return df1

def cmr(df, product='ATL12', release=7):
    # get all cycles and RGTs from dataframe
    df1 = all_tracks(df)
    # build requests for each track
    # this reduces the number of CMR calls but also keeps
    # the API request URL under the character limit
    granule_id, url = [], []
    for tracks, i in df1.groupby('track').groups.items():
        cycles = df1.loc[i, 'cycle'].tolist()
        # build CMR request for RGTs and cycles
        ids, urls = is2tk.utilities.cmr(product=product,
            release=release, cycles=cycles, tracks=tracks,
            provider='NSIDC_CPRD', verbose=False)
        granule_id.extend(ids)
        url.extend(urls)
    # convert to dataframe
    df2 = pd.DataFrame({'granule_id':granule_id, 'url':url})
    # reduce to unique values
    df2 = df2.drop_duplicates()
    # return the dataframe of granule ids and urls
    return df2

def get_granules(df, product='ATL12', release=7, timeout=60):
    # get granule ids and urls from CMR
    df2 = cmr(df, product=product, release=release)
    # create subdirectory for product and release
    subdir = filepath.joinpath(f'{product}.{release:07d}')
    subdir.mkdir(parents=True, exist_ok=True)
    # check that we have each granule
    for i, row in df2.iterrows():
        granule = subdir.joinpath(row['granule_id'])
        if not granule.exists():
            is2tk.utilities.from_nsidc(row['url'], local=granule,
                timeout=timeout, verbose=True)

def find_granules(df, product='ATL12', release=7):
    # get all cycles and RGTs from dataframe
    df1 = all_tracks(df)
    # subdirectory for product and release
    subdir = filepath.joinpath(f'{product}.{release:03d}')
    # build regular expression for finding granules
    tracks = df1.track.astype(str).str.zfill(4)
    cycles = df1.cycle.astype(str).str.zfill(2)
    orbits = r'|'.join(tracks + cycles)
    # regular expression pattern for finding granules
    pattern = rf'{product}_(\d{{14}})_({orbits})\d{{2}}_{release:03d}'
    rx = re.compile(pattern, re.VERBOSE)
    granules = [g for g in subdir.iterdir() if rx.match(g.name)]
    return granules

def read_granule(granule):
    # regular expression pattern for extracting information
    pattern = r'(ATL\d{2})(-\d+)?_(\d{14})_(\d{4})(\d{2})'
    rx = re.compile(pattern, re.VERBOSE)
    PRD, HEM, YYYYMMDDHHMMSS, RGT, CYC = rx.findall(granule.name).pop()
    # read data from granule and concatenate into dataframe
    dataframes = []
    func = getattr(is2tk.io, PRD)
    # read data from each beam
    with h5py.File(granule, 'r') as fileID:
        beams = func.find_beams(fileID, KEEP=True)
        for gtx in beams:
            # initialize dictionary for storing variables
            data = {}
            # extract variables from HDF5 file
            for key,val in mapping[PRD].items():
                data[key] = fileID[gtx][val][:]
                # apply fill values
                if hasattr(fileID[gtx][val], 'fillvalue'):
                    fv = fileID[gtx][val].fillvalue
                    data[key] = np.ma.masked_equal(data[key], fv)
            # get derived variables
            atlas_spot_number = fileID[gtx].attrs['atlas_spot_number']
            data['atlas_spot_number'] = int(atlas_spot_number)
            data['ground_track'] = gtx
            data['track'] = int(RGT)
            data['cycle'] = int(CYC)
            # create dataframe and append to list
            dataframes.append(pd.DataFrame(data))
    # concatenate dataframes for each beam
    df = pd.concat(dataframes, ignore_index=True)
    # return the dataframe
    return df

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Creates a parquet file of ICESat-2 data
            around events in the Tech Ref Table
            """
    )
    # command line parameters
    parser.add_argument('product',
        type=str,
        help='ICESat-2 products to run')
    # ICESat-2 data release
    parser.add_argument('--release','-r',
        type=int, default=7,
        help='ICESat-2 Data Release')
    # event type
    parser.add_argument('--event','-e',
        type=str, default='OceanScan',
        help='Event type in Tech Ref Table')
    # buffer time (seconds)
    parser.add_argument('--buffer','-b',
        type=int, default=300,
        help='Buffer time (seconds) around event')
    # connection timeout and number of retry attempts
    parser.add_argument('--timeout','-T',
        type=int, default=120,
        help='Timeout in seconds for blocking operations')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # read excel file with the tech ref table  
    df = from_excel(pattern=args.event)
    # ensure granules are available
    get_granules(df,
        product=args.product,
        release=args.release,
        timeout=args.timeout
    )
    # for each event
    for i, row in df.iterrows():
        # find granules for event
        granules = find_granules(row,
            product=args.product,
            release=args.release
        )
        # read data from each granule and concatenate into dataframe
        dataframes = [read_granule(g) for g in granules]
        # skip event if no data
        if not dataframes:
            continue
        df1 = pd.concat(dataframes, ignore_index=True)
        # filter to time period around event
        delta_time_start = row['delta_time_start'] - args.buffer
        delta_time_end = row['delta_time_end'] + args.buffer
        mask = (df1['delta_time'] >= delta_time_start) & \
               (df1['delta_time'] <= delta_time_end)
        # skip if no data in (buffered) time range
        if not np.any(mask):
            continue
        # filter dataframe to bounds around event
        df2 = df1.loc[mask,:].reset_index(drop=True)
        # write dataframe to output parquet file
        outfile = (f'{args.product}_{args.event}_Cycle{row.cycle:02.0f}_'
            f'RGT{row.rgt_start:04.0f}-{row.rgt_end:04.0f}.parquet')
        df2.to_parquet(filepath.joinpath(outfile), index=False)

if __name__ == '__main__':
    main()
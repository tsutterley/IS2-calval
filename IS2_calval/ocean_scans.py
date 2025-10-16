#!/usr/bin/env python
"""
ocean_scans.py
Written by Tyler Sutterley (10/2025)
Check ATL12 data before and after ocean scans

Tech Ref Table:
https://doi.org/10.5281/zenodo.16283560
"""
import re
import timescale
import numpy as np
import pandas as pd
import icesat2_toolkit as is2tk
import utilities

# get path to data directory
filepath = utilities.get_data_path(['data'])

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
    # build lists with all cycles and RGTs
    cycles, tracks = [], []
    # for each row in the dataframe
    for row in df.itertuples():
        # list of RGTs for row
        RGT = [r for r in range(row.rgt_start, row.rgt_end+1)]
        # cycle as string
        cycle = [row.cycle]*len(RGT)
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

def get_granules(df):
    # get granule ids and urls from CMR
    df2 = cmr(df, product='ATL12', release=7)
    # check that we have each granule
    for i, row in df2.iterrows():
        granule = filepath.joinpath(row['granule_id'])
        if not granule.exists():
            is2tk.utilities.from_nsidc(row['url'], local=granule,
                timeout=60, verbose=True)

if __name__ == '__main__':
    df = from_excel()
    get_granules(df)

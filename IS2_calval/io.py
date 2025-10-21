#!/usr/bin/env python
"""
io.py
Written by Tyler Sutterley (10/2025)
Reads a subset of variables from an ICESat-2 HDF5 file

UPDATE HISTORY:
    Written 10/2025
"""
import re
import h5py
import pathlib
import numpy as np
import pandas as pd
import icesat2_toolkit as is2tk

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
    h_var = 'ssh_segments/heights/h_var',
    geoid_free2mean_seg = 'ssh_segments/stats/geoid_free2mean_seg',
    geoid_seg = 'ssh_segments/stats/geoid_seg',
    ice_conc = 'ssh_segments/stats/ice_conc',
    length_seg = 'ssh_segments/heights/length_seg',
    n_photon = 'ssh_segments/stats/n_photons',
    n_pulse_seg = 'ssh_segments/heights/n_pulse_seg',
    n_ttl_photon = 'ssh_segments/stats/n_ttl_photon',
    near_sat_fract_seg = 'ssh_segments/stats/near_sat_fract_seg',
    swh = 'ssh_segments/heights/swh'
)

def read_granule(granule, **kwargs):
    """
    Reads a subset of variables from an ICESat-2 HDF5 file
    """
    kwargs.setdefault('field_mapping', {})
    # verify path to granule
    granule = pathlib.Path(granule).expanduser().absolute()
    assert granule.exists(), f'Granule not found: {granule}'
    # regular expression pattern for extracting information
    pattern = r'(ATL\d{2})(-\d+)?_(\d{14})_(\d{4})(\d{2})'
    rx = re.compile(pattern, re.VERBOSE)
    PRD, HEM, YYYYMMDDHHMMSS, RGT, CYC = rx.findall(granule.name).pop()
    # read data from granule and concatenate into dataframe
    dataframes = []
    func = getattr(is2tk.io, PRD)
    # merge variable mapping
    field_mapping = mapping[PRD].copy()
    field_mapping.update(kwargs['field_mapping'])
    # read data from each beam
    with h5py.File(granule, 'r') as fileID:
        beams = func.find_beams(fileID, KEEP=True)
        for gtx in beams:
            # initialize dictionary for storing variables
            data = {}
            # extract variables from HDF5 file
            for key,val in field_mapping.items():
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

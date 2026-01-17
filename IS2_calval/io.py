#!/usr/bin/env python
"""
io.py
Written by Tyler Sutterley (10/2025)
Reads a subset of variables from an ICESat-2 HDF5 file

UPDATE HISTORY:
    Updated 01/2026: added some basic ATL03 functions
    Written 10/2025
"""

import re
import h5py
import pathlib
import numpy as np
import pandas as pd

# variable mapping
mapping = {}
mapping["ATL03"] = dict(
    delta_time="geophys_corr/delta_time",
    reference_photon_lon="geolocation/reference_photon_lon",
    reference_photon_lat="geolocation/reference_photon_lat",
    geoid="geophys_corr/geoid",
    geoid_free2mean="geophys_corr/geoid_free2mean",
)
mapping["ATL12"] = dict(
    delta_time="ssh_segments/delta_time",
    longitude="ssh_segments/longitude",
    latitude="ssh_segments/latitude",
    fpb_corr="ssh_segments/heights/fpb_corr",
    h="ssh_segments/heights/h",
    h_ice_free="ssh_segments/heights/h_ice_free",
    h_ice_free_uncrtn="ssh_segments/heights/h_ice_free_uncrtn",
    h_var="ssh_segments/heights/h_var",
    geoid_free2mean_seg="ssh_segments/stats/geoid_free2mean_seg",
    geoid_seg="ssh_segments/stats/geoid_seg",
    ice_conc="ssh_segments/stats/ice_conc",
    length_seg="ssh_segments/heights/length_seg",
    n_photon="ssh_segments/stats/n_photons",
    n_pulse_seg="ssh_segments/heights/n_pulse_seg",
    n_ttl_photon="ssh_segments/stats/n_ttl_photon",
    near_sat_fract_seg="ssh_segments/stats/near_sat_fract_seg",
    orbit_number="ssh_segments/stats/orbit_number",
    swh="ssh_segments/heights/swh",
)


def find_beams(fid, product="ATL12", pattern=r"gt\d[lr]"):
    """
    Find beam groups within a file

    Parameters
    ----------
    fid: h5py.File
        Open HDF5 file object
    product: str
        ICESat-2 product
    pattern: str
        Regular expression pattern for identifying beams

    Returns
    -------
    beams: list
        List of beam groups within the file
    """
    # list of beams
    beams = []
    # variable to check
    val = mapping[product]["delta_time"]
    # read each input beam within the file
    for gtx in [k for k in fid.keys() if bool(re.match(pattern, k))]:
        # check if subsetted beam contains time data
        try:
            fid[gtx][val]
        except KeyError:
            pass
        else:
            beams.append(gtx)
    return beams


def orbit_number_to_track(orbit_number: np.ndarray):
    """
    Convert orbit number to reference ground track (RGT)

    Parameters
    ----------
    orbit_number: np.ndarray
        Orbit number(s) to convert
    """
    # number of orbits per cycle
    orbits_per_cycle = 1387
    return np.mod(orbit_number - 201, orbits_per_cycle)


def read_granule(granule, **kwargs):
    """
    Reads a subset of variables from an ICESat-2 HDF5 file

    Parameters
    ----------
    granule: str or pathlib.Path
        Path to the ICESat-2 granule
    field_mapping: dict
        Dictionary mapping of variable names to HDF5 paths

    Returns
    -------
    df: pandas.DataFrame
        DataFrame of variables from the granule
    """
    kwargs.setdefault("field_mapping", {})
    # verify path to granule
    granule = pathlib.Path(granule).expanduser().absolute()
    assert granule.exists(), f"Granule not found: {granule}"
    # regular expression pattern for extracting information
    pattern = r"(ATL\d{2})(-\d+)?_(\d{14})_(\d{4})(\d{2})"
    rx = re.compile(pattern, re.VERBOSE)
    PRD, HEM, YYYYMMDDHHMMSS, RGT, CYC = rx.findall(granule.name).pop()
    # read data from granule and concatenate into dataframe
    dataframes = []
    # merge variable mapping
    field_mapping = mapping[PRD].copy()
    field_mapping.update(kwargs["field_mapping"])
    # read data from each beam
    with h5py.File(granule, "r") as fid:
        beams = find_beams(fid, product=PRD)
        for gtx in beams:
            # initialize dictionary for storing variables
            data = {}
            # extract variables from HDF5 file
            for key, val in field_mapping.items():
                # attempt to read variable
                try:
                    data[key] = fid[gtx][val][:]
                except KeyError:
                    continue
                # apply fill values
                if hasattr(fid[gtx][val], "fillvalue"):
                    fv = fid[gtx][val].fillvalue
                    data[key] = np.ma.masked_equal(data[key], fv)
            # get derived variables
            atlas_spot_number = fid[gtx].attrs["atlas_spot_number"]
            data["atlas_spot_number"] = int(atlas_spot_number)
            data["ground_track"] = gtx
            data["track"] = int(RGT)
            data["cycle"] = int(CYC)
            # create dataframe and append to list
            dataframes.append(pd.DataFrame(data))
    # concatenate dataframes for each beam
    df = pd.concat(dataframes, ignore_index=True)
    # return the dataframe
    return df


def reference_photon_height(granule, gtx, minimum_weight=0):
    """
    Extract the height of a height of a reference photon

    Parameters
    ----------
    granule: str or pathlib.Path
        Path to the ATL03 granule
    gtx: str
        Beam group within the granule
    minimum_weight: int
        Minimum weight for reference photon selection

    Returns
    -------
    height: np.ndarray
        Height of the reference photons
    """
    # open ATL03 granule
    with h5py.File(granule, "r") as fid:
        # extract index and mapping variables
        reference_photon_index = (
            fid[gtx]["geolocation"]["reference_photon_index"][:] - 1
        )
        ph_index_beg = fid[gtx]["geolocation"]["ph_index_beg"][:] - 1
        # mask for valid segments
        valid = reference_photon_index >= 0
        # calculate photon index (convert to 0-based index)
        photon_index = ph_index_beg[valid] + reference_photon_index[valid]
        # extract output heights
        h_ph = fid[gtx]["heights"]["h_ph"][:]
        height = np.full_like(ph_index_beg, np.nan, dtype=h_ph.dtype)
        height[valid] = h_ph[photon_index]
        # verify quality and weight
        quality_ph = fid[gtx]["heights"]["quality_ph"][:]
        weight_ph = fid[gtx]["heights"]["weight_ph"][:]
        # apply quality and weight mask
        height[valid] = np.where(
            (
                (quality_ph[photon_index] == 0)
                & (weight_ph[photon_index] >= minimum_weight)
            ),
            height[valid],
            np.nan,
        )
    # return the reference photon heights
    return height


def is_surface_type(granule, gtx, column=1, exclusive=True):
    """
    Check if an ATL03 segment is a surface type

    Parameters
    ----------
    granule: str or pathlib.Path
        Path to the ATL03 granule
    gtx: str
        Beam group within the granule
    column: int or list
        Column index or list of indices for surface type
        0: land
        1: ocean
        2: sea ice
        3: land ice
        4: inland water
    exclusive: bool
        Only return segments that are exclusively the specified type(s)

    Returns
    -------
    is_type: np.ndarray
        Boolean mask of segments
    """
    # open ATL03 granule
    with h5py.File(granule, "r") as fid:
        surf_type = fid[gtx]["geolocation"]["surf_type"][:].astype(bool)
    # initialize masks
    ds_time, ds_surf_type = surf_type.shape
    not_type = np.zeros((ds_time), dtype=bool)
    # convert column to list if integer
    if isinstance(column, int):
        column = [column]
    # iterate over surface types
    # 0: land
    # 1: ocean
    # 2: sea ice
    # 3: land ice
    # 4: inland water
    for i in range(ds_surf_type):
        if i in column:
            is_type = surf_type[:, i].copy()
        else:
            not_type |= surf_type[:, i]
    # return mask for surface type
    if exclusive:
        return is_type & np.logical_not(not_type)
    else:
        return is_type

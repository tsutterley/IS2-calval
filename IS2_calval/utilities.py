#!/usr/bin/env python
u"""
utilities.py
Written by Tyler Sutterley (10/2025)
Download and management utilities for syncing time and auxiliary files

PYTHON DEPENDENCIES:
    lxml: processing XML and HTML in Python
        https://pypi.python.org/pypi/lxml

UPDATE HISTORY:
    Written 10/2025
"""
from __future__ import annotations

import io
import re
import sys
import ssl
import json
import shutil
import inspect
import logging
import pathlib
import hashlib
import zipfile
import importlib
import posixpath
import lxml.etree
if sys.version_info[0] == 2:
    from urllib import quote_plus
    from cookielib import CookieJar
    import urllib2
else:
    from urllib.parse import quote_plus
    from http.cookiejar import CookieJar
    import urllib.request as urllib2

# PURPOSE: get absolute path within a package from a relative path
def get_data_path(relpath: list | str | pathlib.Path):
    """
    Get the absolute path within a package from a relative path

    Parameters
    ----------
    relpath: list, str or pathlib.Path
        relative path
    """
    # current file path
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    filepath = pathlib.Path(filename).absolute().parent
    if isinstance(relpath, list):
        # use *splat operator to extract from list
        return filepath.joinpath(*relpath)
    elif isinstance(relpath, (str, pathlib.Path)):
        return filepath.joinpath(relpath)

def import_dependency(
        name: str,
        extra: str = "",
        raise_exception: bool = False
    ):
    """
    Import an optional dependency

    Adapted from ``pandas.compat._optional::import_optional_dependency``

    Parameters
    ----------
    name: str
        Module name
    extra: str, default ""
        Additional text to include in the ``ImportError`` message
    raise_exception: bool, default False
        Raise an ``ImportError`` if the module is not found

    Returns
    -------
    module: obj
        Imported module
    """
    # check if the module name is a string
    msg = f"Invalid module name: '{name}'; must be a string"
    assert isinstance(name, str), msg
    # default error if module cannot be imported
    err = f"Missing optional dependency '{name}'. {extra}"
    module = type('module', (), {})
    # try to import the module
    try:
        module = importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as exc:
        if raise_exception:
            raise ImportError(err) from exc
        else:
            logging.debug(err)
    # return the module
    return module

# PURPOSE: get the sheet names from an Excel file
def get_excel_sheet_names(xls_file=None, pattern=None):
    """
    Get the sheet names from an Excel file
    
    Parameters
    ----------
    xls_file: str, pathlib.Path or BytesIO, default None
        path to local Excel file or BytesIO object
    pattern: str or NoneType, default None
        regular expression pattern to filter sheet names
    """
    # compile xml parsers for lxml
    parser = lxml.etree.XMLParser(recover=True, remove_blank_text=True)
    with zipfile.ZipFile(xls_file, 'r') as z:
        tree = lxml.etree.parse(z.open("xl/workbook.xml"), parser)
    # get the XML root
    root = tree.getroot()
    # find all the sheet names
    sheets = root.find('sheets', root.nsmap)
    names = [s.get('name') for s in sheets.findall('sheet', root.nsmap)]
    # filter names by pattern
    if pattern is not None:
        names = [name for name in names if re.search(pattern, name, re.I)]
    # return the list of sheet names
    return names

# PURPOSE: zenodo record for the tech ref table
def get_zenodo_url(zenodo_record='16283560'):
    """
    Get the zenodo url and checksum for a record
    
    Parameters
    ----------
    zenodo_record: str, default '16283560'
        zenodo record number
    """
    zenodo = 'https://zenodo.org/api'
    records_api = f'{zenodo}/records/{zenodo_record}'
    version_record = from_json(records_api)['id']
    deposit_api = f'{zenodo}/deposit/depositions/{version_record}/files'
    response = from_json(deposit_api)
    download = response[0]['links']['download']
    checksum = response[0]['checksum']
    return download, checksum

# PURPOSE: recursively split a url path
def url_split(s: str):
    """
    Recursively split a url path into a list

    Parameters
    ----------
    s: str
        url string
    """
    head, tail = posixpath.split(s)
    if head in ('http:','https:','ftp:','s3:'):
        return s,
    elif head in ('', posixpath.sep):
        return tail,
    return url_split(head) + (tail,)

def _create_default_ssl_context() -> ssl.SSLContext:
    """Creates the default SSL context
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    _set_ssl_context_options(context)
    context.options |= ssl.OP_NO_COMPRESSION
    return context

def _create_ssl_context_no_verify() -> ssl.SSLContext:
    """Creates an SSL context for unverified connections
    """
    context = _create_default_ssl_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

def _set_ssl_context_options(context: ssl.SSLContext) -> None:
    """Sets the default options for the SSL context
    """
    if sys.version_info >= (3, 10) or ssl.OPENSSL_VERSION_INFO >= (1, 1, 0, 7):
        context.minimum_version = ssl.TLSVersion.TLSv1_2
    else:
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1

# default ssl context
_default_ssl_context = _create_ssl_context_no_verify()

# PURPOSE: download a file from a http host
def from_http(
        HOST: str | list,
        timeout: int | None = None,
        context = _default_ssl_context,
        hash: str = '',
        chunk: int = 16384,
    ):
    """
    Download a file from a http host

    Parameters
    ----------
    HOST: str or list
        remote http host path split as list
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default ssl.SSLContext(ssl.PROTOCOL_TLS)
        SSL context for ``urllib`` opener object
    local: str, pathlib.Path or NoneType, default None
        path to local file
    hash: str, default ''
        MD5 hash of local file
    chunk: int, default 16384
        chunk size for transfer encoding

    Returns
    -------
    remote_buffer: obj
        BytesIO representation of file
    """
    # verify inputs for remote http host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    # try downloading from http
    try:
        # Create and submit request.
        request = urllib2.Request(posixpath.join(*HOST))
        response = urllib2.urlopen(request, timeout=timeout, context=context)
    except (urllib2.HTTPError, urllib2.URLError) as exc:
        raise Exception('Download error from {0}'.format(posixpath.join(*HOST)))
    else:
        # copy remote file contents to bytesIO object
        remote_buffer = io.BytesIO()
        shutil.copyfileobj(response, remote_buffer, chunk)
        remote_buffer.seek(0)
        # save file basename with bytesIO object
        remote_buffer.filename = HOST[-1]
        # generate checksum hash for remote file
        remote_hash = hashlib.md5(remote_buffer.getvalue()).hexdigest()
        # compare hashes if provided
        assert (hash == remote_hash) if hash else True
        # return the bytesIO object
        remote_buffer.seek(0)
        return remote_buffer

# PURPOSE: load a JSON response from a http host
def from_json(
        HOST: str | list,
        timeout: int | None = None,
        context: ssl.SSLContext = _default_ssl_context
    ) -> dict:
    """
    Load a JSON response from a http host

    Parameters
    ----------
    HOST: str or list
        remote http host path split as list
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default pyTMD.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    """
    # verify inputs for remote http host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    # try loading JSON from http
    try:
        # Create and submit request for JSON response
        request = urllib2.Request(posixpath.join(*HOST))
        request.add_header('Accept', 'application/json')
        response = urllib2.urlopen(request, timeout=timeout, context=context)
    except urllib2.HTTPError as exc:
        logging.debug(exc.code)
        raise RuntimeError(exc.reason) from exc
    except urllib2.URLError as exc:
        logging.debug(exc.reason)
        msg = 'Load error from {0}'.format(posixpath.join(*HOST))
        raise Exception(msg) from exc
    else:
        # load JSON response
        return json.loads(response.read())

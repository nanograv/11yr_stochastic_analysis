from __future__ import division, print_function
import numpy as np
from glob import glob
import json

from enterprise.pulsar import Pulsar

DATADIR = 'nano11y_data/'

def get_pulsars(psrlist=None, ephem='DE436'):
    """
    Reads in list of pulsar names and ephemeris version
    and returns list of instantiated enterprise Pulsar objects.

    By default the input list is None and we use the 34 pulsars used in
    the stochastic background analysis. The default ephemeris is that used
    in the 11-year dataset (DE436)
    """
    if psrlist is None:
        psrlist = np.loadtxt('psrlist.txt', dtype='S32')

    parfiles = sorted(glob(DATADIR+'/partim/*.par'))
    timfiles = sorted(glob(DATADIR+'/partim/*.tim'))

    psrs = []
    for par, tim in zip(parfiles, timfiles):
        pname = par.split('/')[-1].split('_')[0]
        if pname in psrlist:
            psrs.append(Pulsar(par, tim, ephem=ephem))

    return psrs

def get_noise_dict(psrlist=None):
    """
    Reads in list of pulsar names and returns dictionary
    of {parameter_name: value} for all noise parameters.

    By default the input list is None and we use the 34 pulsars used in
    the stochastic background analysis.
    """

    if psrlist is None:
        psrlist = np.loadtxt('psrlist.txt', dtype='S32')

    params = {}
    for p in psrlist:
        with open(DATADIR+'/noisefiles/{}_noise.json'.format(p), 'r') as fin:
            params.update(json.load(fin))

    return params

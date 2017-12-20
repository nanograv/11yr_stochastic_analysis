from __future__ import division, print_function
import numpy as np

from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import utils

from utils import get_noise_dict


def model_1(psr):
    """
    Reads in enterprise Pulsar instance and returns a PTA
    instantiated with the standard NANOGrav noise model:

        1. EFAC per backend/receiver system
        2. EQUAD per backend/receiver system
        3. ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.
    """

    # define selection by observing backend
    selection = selections.Selection(selections.by_backend)

    # white noise parameters
    efac = parameter.Uniform(0.01, 10.0)
    equad = parameter.Uniform(-8.5, -5)
    ecorr = parameter.Uniform(-8.5, -5)

    # red noise parameters
    log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)

    # white noise signals
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
    eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

    # red noise signal
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30)

    # timing model
    tm = gp_signals.TimingModel()

    # full model
    s = ef + eq + ec + tm + rn

    # set up PTA of one
    pta = signal_base.PTA([s(psr)])

    return pta


def model_2a(psrs, gamma_common=None, upper_limit=False, bayesephem=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2A from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. Common red noise process modeled as power-law with 30
        sampling frequencies

    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    """

    # find the maximum time span to set GW frequency sampling
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    # define selection by observing backend
    selection = selections.Selection(selections.by_backend)

    # white noise parameters
    efac = parameter.Constant()
    equad = parameter.Constant()
    ecorr = parameter.Constant()

    # red noise parameters
    if upper_limit:
        log10_A = parameter.LinearExp(-20, -11)
    else:
        log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)

    # common red noise parameters
    if upper_limit:
        log10_Agw = parameter.LinearExp(-18, -11)('log10_A_gw')
    elif not upper_limit and gamma_common is not None:
        if np.abs(gamma_common - 4.33) < 0.1:
            log10_Agw = parameter.Uniform(-18, -14)('log10_A_gw')
        else:
            log10_Agw = parameter.Uniform(-18, -11)('log10_A_gw')
    else:
        log10_Agw = parameter.Uniform(-18, -11)('log10_A_gw')

    if gamma_common is not None:
        gamma_gw = parameter.Constant(gamma_common)
    else:
        gamma_gw = parameter.Uniform(0, 7)('gamma_gw')

    # white noise signals
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
    eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

    # red noise signal
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=Tspan)

    # common red noise signal
    cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
    crn = gp_signals.FourierBasisGP(cpl, components=30, Tspan=Tspan)

    # ephemeris model
    if bayesephem:
        eph = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    tm = gp_signals.TimingModel()

    # full model
    s = ef + eq + ec + tm + rn + crn
    if bayesephem:
        s += eph

    # set up PTA
    pta = signal_base.PTA([s(psr) for psr in psrs])

    # set white noise parameters
    noisedict = get_noise_dict(psrlist=[p.name for p in psrs])
    pta.set_default_params(noisedict)

    return pta

def model_3a(psrs, gamma_common=None, upper_limit=False, bayesephem=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 3A from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. GWB with HD correlations modeled as power-law with 30 sampling
        frequencies

    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    """

    # find the maximum time span to set GW frequency sampling
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    # define selection by observing backend
    selection = selections.Selection(selections.by_backend)

    # white noise parameters
    efac = parameter.Constant()
    equad = parameter.Constant()
    ecorr = parameter.Constant()

    # red noise parameters
    if upper_limit:
        log10_A = parameter.LinearExp(-20, -11)
    else:
        log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)

    # common red noise parameters
    if upper_limit:
        log10_Agw = parameter.LinearExp(-18, -11)('log10_A_gw')
    elif not upper_limit and gamma_common is not None:
        if np.abs(gamma_common - 4.33) < 0.1:
            log10_Agw = parameter.Uniform(-18, -14)('log10_A_gw')
        else:
            log10_Agw = parameter.Uniform(-18, -11)('log10_A_gw')
    else:
        log10_Agw = parameter.Uniform(-18, -11)('log10_A_gw')

    if gamma_common is not None:
        gamma_gw = parameter.Constant(gamma_common)
    else:
        gamma_gw = parameter.Uniform(0, 7)('gamma_gw')

    # white noise signals
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
    eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

    # red noise signal
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=Tspan)

    # common red noise signal
    cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
    orf = utils.hd_orf()
    crn = gp_signals.FourierBasisCommonGP(cpl, orf, components=30, Tspan=Tspan)

    # ephemeris model
    if bayesephem:
        eph = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    tm = gp_signals.TimingModel()

    # full model
    s = ef + eq + ec + tm + rn + crn
    if bayesephem:
        s += eph

    # set up PTA
    pta = signal_base.PTA([s(psr) for psr in psrs])

    # set white noise parameters
    noisedict = get_noise_dict(psrlist=[p.name for p in psrs])
    pta.set_default_params(noisedict)

    return pta

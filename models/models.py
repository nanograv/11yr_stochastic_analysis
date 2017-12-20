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

#### Model component building blocks ####

def white_noise_block(vary=False):
    """
    Returns the white noise block of the model:

        1. EFAC per backend/receiver system
        2. EQUAD per backend/receiver system
        3. ECORR per backend/receiver system

    :param vary:
        If set to true we vary these parameters
        with uniform priors. Otherwise they are set to constants
        with values to be set later.
    """

    # define selection by observing backend
    selection = selections.Selection(selections.by_backend)

    # white noise parameters
    if vary:
        efac = parameter.Uniform(0.01, 10.0)
        equad = parameter.Uniform(-8.5, -5)
        ecorr = parameter.Uniform(-8.5, -5)
    else:
        efac = parameter.Constant()
        equad = parameter.Constant()
        ecorr = parameter.Constant()

    # white noise signals
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
    eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

    # combine signals
    s = ef + eq + ec

    return s


def red_noise_block(prior='log-uniform', Tspan=None):
    """
    Returns red noise model:

        1. Red noise modeled as a power-law with 30 sampling frequencies

    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for indivicual pulsar.

    """

    # red noise parameters
    if prior == 'uniform':
        log10_A = parameter.LinearExp(-20, -11)
    elif prior == 'log-uniform':
        log10_A = parameter.Uniform(-20, -11)
    else:
        raise ValueError('Unknown prior for red noise amplitude!')

    gamma = parameter.Uniform(0, 7)

    # red noise signal
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=Tspan)

    return rn


def common_red_noise_block(prior='log-uniform', Tspan=None, gamma_val=None, orf=None):
    """
    Returns common red noise model:

        1. Red noise modeled as a power-law with 30 sampling frequencies

    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for indivicual pulsar.
    :param gamma_val:
        Value of spectral index for power-law and turnover
        models. By default spectral index is varied of range [0,7]
    :param orf:
        String representing which overlap reduction function to use.
        By default we do not use any spatial correlations. Permitted
        values are ['hd', 'dipole', 'monopole'].

    """

    # common red noise parameters
    if prior == 'uniform':
        log10_Agw = parameter.LinearExp(-18, -11)('log10_A_gw')
    elif prior == 'log-uniform' and gamma_val is not None:
        if np.abs(gamma_val - 4.33) < 0.1:
            log10_Agw = parameter.Uniform(-18, -14)('log10_A_gw')
        else:
            log10_Agw = parameter.Uniform(-18, -11)('log10_A_gw')
    else:
        log10_Agw = parameter.Uniform(-18, -11)('log10_A_gw')

    if gamma_val is not None:
        gamma_gw = parameter.Constant(gamma_val)
    else:
        gamma_gw = parameter.Uniform(0, 7)('gamma_gw')

    # common red noise signal
    cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
    if orf is None:
        crn = gp_signals.FourierBasisGP(cpl, components=30, Tspan=Tspan)
    elif orf == 'hd':
        orf = utils.hd_orf()
        crn = gp_signals.FourierBasisCommonGP(cpl, orf, components=30, Tspan=Tspan)
    else:
        raise ValueError('ORF {} not recognized'.format(orf))

    return crn

#### PTA models from paper ####
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

    # white noise
    s = white_noise_block(vary=True)

    # red noise
    s += red_noise_block()

    # timing model
    s += gp_signals.TimingModel()

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
        2. Optional physical ephemeris modeling.

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

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    # white noise
    s = white_noise_block(vary=False)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan)

    # common red noise block
    s += common_red_noise_block(prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

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
        2. Optional physical ephemeris modeling.

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

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    # white noise
    s = white_noise_block(vary=False)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan)

    # common red noise block
    s += common_red_noise_block(prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, orf='hd')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # timing model
    s += gp_signals.TimingModel()

    # set up PTA
    pta = signal_base.PTA([s(psr) for psr in psrs])

    # set white noise parameters
    noisedict = get_noise_dict(psrlist=[p.name for p in psrs])
    pta.set_default_params(noisedict)

    return pta

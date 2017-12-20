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

#### Extra model componentents not part of base enterprise ####

@signal_base.function
def free_spectrum(f, log10_rho=None):
    """
    Free spectral model. PSD  amplitude at each frequency
    is a free parameter. Model is parameterized by
    S(f_i) = \rho_i^2 * T,
    where \rho_i is the free parameter and T is the observation
    length.
    """
    return np.repeat(10**(2*log10_rho), 2)

@signal_base.function
def dipole_orf(pos1, pos2):
    """Dipole spatial correlation function."""
    if np.all(pos1 == pos2):
        return 1
    else:
        return 0.5 * np.dot(pos1, pos2)

@signal_base.function
def monopole_orf(pos1, pos2):
    """Monopole spatial correlation function."""
    if np.all(pos1 == pos2):
        return 1
    else:
        return 0.5


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


def common_red_noise_block(psd='powerlaw', prior='log-uniform',
                           Tspan=None, gamma_val=None, orf=None,
                           name='gw'):
    """
    Returns common red noise model:

        1. Red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']
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
    :param name: Name of common red process

    """

    orfs = {'hd': utils.hd_orf(), 'dipole': dipole_orf(),
            'monopole': monopole_orf()}

    # common red noise parameters
    if psd in ['powerlaw', 'turnover']:
        amp_name = 'log10_A_{}'.format(name)
        if prior == 'uniform':
            log10_Agw = parameter.LinearExp(-18, -11)(amp_name)
        elif prior == 'log-uniform' and gamma_val is not None:
            if np.abs(gamma_val - 4.33) < 0.1:
                log10_Agw = parameter.Uniform(-18, -14)(amp_name)
            else:
                log10_Agw = parameter.Uniform(-18, -11)(amp_name)
        else:
            log10_Agw = parameter.Uniform(-18, -11)(amp_name)

        gam_name = 'gamma_{}'.format(name)
        if gamma_val is not None:
            gamma_gw = parameter.Constant(gamma_val)
        else:
            gamma_gw = parameter.Uniform(0, 7)(gam_name)

        # common red noise PSD
        if psd == 'powerlaw':
            cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        elif psd == 'turnover':
            kappa_name = 'kappa_{}'.format(name)
            lf0_name = 'log10_fbend_{}'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lf0_gw = parameter.Uniform(-9, -7)(lf0_name)
            cpl = utils.turnover(log10_A=log10_Agw, gamma=gamma_gw,
                                 lf0=lf0_gw, kappa=kappa_gw)

    if psd == 'spectrum':
        rho_name = 'log10_rho_{}'.format(name)
        if prior == 'uniform':
            log10_rho_gw = parameter.LinearExp(-9, -4, size=30)(rho_name)
        elif prior == 'log-uniform':
            log10_rho_gw = parameter.Uniform(-9, -4, size=30)(rho_name)

        cpl = free_spectrum(log10_rho=log10_rho_gw)

    if orf is None:
        crn = gp_signals.FourierBasisGP(cpl, components=30, Tspan=Tspan)
    elif orf in orfs.keys():
        crn = gp_signals.FourierBasisCommonGP(cpl, orfs[orf], components=30, Tspan=Tspan)
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


def model_2a(psrs, psd='powerlaw', gamma_common=None,
             upper_limit=False, bayesephem=False):
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
        1.Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
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
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, name='gw')

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


def model_2b(psrs, psd='powerlaw', gamma_common=None,
             upper_limit=False, bayesephem=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2B from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. Dipole spatially correlated signal modeled with PSD.
        Default PSD is powerlaw. Available options
        ['powerlaw', 'turnover', 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
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

    # dipole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, orf='dipole',
                                name='dipole')

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


def model_2c(psrs, psd='powerlaw', gamma_common=None,
             upper_limit=False, bayesephem=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2C from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. Dipole spatially correlated signal modeled with PSD.
        Default PSD is powerlaw. Available options
        ['powerlaw', 'turnover', 'spectrum']
        2. Monopole spatially correlated signal modeled with PSD.
        Default PSD is powerlaw. Available options
        ['powerlaw', 'turnover', 'spectrum']
        3. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
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

    # dipole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, orf='dipole',
                                name='dipole')

    # monopole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, orf='monopole',
                                name='monopole')

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


def model_2d(psrs, psd='powerlaw', gamma_common=None,
             upper_limit=False, bayesephem=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2D from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. Monopole spatially correlated signal modeled with PSD.
        Default PSD is powerlaw. Available options
        ['powerlaw', 'turnover', 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
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

    # monopole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, orf='monopole',
                                name='monopole')

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


def model_3a(psrs, psd='powerlaw', gamma_common=None, upper_limit=False, bayesephem=False):
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
        1. GWB with HD correlations modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum'] 'powerlaw' is default
        value.
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
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, orf='hd', name='gw')

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

def model_3b(psrs, psd='powerlaw', gamma_common=None, upper_limit=False, bayesephem=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 3B from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. GWB with HD correlations modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Dipole signal modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        3. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum'] 'powerlaw' is default
        value.
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
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, orf='hd', name='gw')

    # dipole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, orf='dipole',
                                name='dipole')

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

def model_3c(psrs, psd='powerlaw', gamma_common=None, upper_limit=False, bayesephem=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 3C from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. GWB with HD correlations modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Dipole signal modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        3. Monopole signal modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        4. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum'] 'powerlaw' is default
        value.
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
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, orf='hd', name='gw')

    # dipole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, orf='dipole',
                                name='dipole')

    # monopole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, orf='monopole',
                                name='monopole')

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


def model_3d(psrs, psd='powerlaw', gamma_common=None, upper_limit=False, bayesephem=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 3D from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. GWB with HD correlations modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Monopole signal modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        3. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum'] 'powerlaw' is default
        value.
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
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, orf='hd', name='gw')

    # monopole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                gamma_val=gamma_common, orf='monopole',
                                name='monopole')

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
